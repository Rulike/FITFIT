(function () {
  'use strict';

  const API = window.EventAPI;
  const Charts = window.FinCharts;
  const COLORS = Object.freeze({
    primary: '#2ee6a6',
    blue: '#4aa8ff',
    purple: '#9677ff',
    violet: '#9677ff',
    warning: '#ffb648',
    danger: '#ff5d73',
    cyan: '#4ce3ef',
  });
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const preferences = {
    get(key, fallback) { try { return localStorage.getItem(key) || fallback; } catch (_) { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch (_) { /* Local-file privacy mode may disable storage. */ } }
  };

  const routes = {
    dashboard: { label: '概览看板', eyebrow: 'OVERVIEW', icon: '▥', section: '核心功能' },
    events: { label: '事件监测', eyebrow: 'EVENT MONITORING', icon: '◎', section: '核心功能' },
    credibility: { label: '可信评估', eyebrow: 'CREDIBILITY', icon: '♢', section: '核心功能' },
    timeline: { label: '事件脉络', eyebrow: 'EVENT TIMELINE', icon: '⌁', section: '核心功能' },
    alerts: { label: '预警中心', eyebrow: 'ALERT CENTER', icon: '♧', section: '核心功能', badge: 0 },
    graph: { label: '知识图谱', eyebrow: 'KNOWLEDGE GRAPH', icon: '⌬', section: '核心功能' },
    integration: { label: '任务链路回放', eyebrow: 'EVENT PIPELINE', icon: '◈', section: '分析视图' }
  };

  const PROCESS_STAGE_KEYS = ['ingest', 'task1', 'task2', 'task3', 'task4'];

  const state = {
    route: 'dashboard', eventPage: 1, eventFilters: { query: '', level: 'all', sentiment: 'all' }, selectedEvent: '',
    alertFilters: { level: 'all', status: 'all', eventId: '' }, alertRule: { critical: 90, high: 75, credibility: 0, channel: '平台消息' }, graphScope: 'company', graphController: null, theme: preferences.get('finsight-theme', 'light'),
    processDemo: { data: null, activeIndex: -1, stageIndex: -1, completedStage: -1, processingStage: -1, processingEntryId: '', timer: null, runToken: 0 }, navigationId: 0, eventRequest: 0, alertRequest: 0, searchRequest: 0, drawerRequest: 0, searchMode: 'preview'
  };

  function escapeHTML(value = '') {
    return String(value).replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
  }

  function formatNumber(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? new Intl.NumberFormat('en-US').format(Math.round(numeric)) : '0';
  }

  function levelLabel(level) {
    return ({ critical: '重大', high: '高影响', medium: '中影响', low: '低影响' })[level] || level;
  }

  function sentimentLabel(sentiment) {
    return ({ positive: '正面', negative: '负面', neutral: '中性' })[sentiment] || sentiment;
  }

  function formatPercent(value, digits = 2) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return `0.${'0'.repeat(digits)}%`;
    const scaled = numeric <= 1 ? numeric * 100 : numeric;
    return `${scaled.toFixed(digits)}%`;
  }

  function task3StateLabelLocal(stateType) {
    return ({
      unknown: '待判定',
      announcement: '公告确认',
      completion: '完成',
	      progress: '推进中',
	      plan: '计划中',
	      confirmation_proposal: '确认提议',
	      investigation: '调查中',
	      refutation: '反转核验',
	      penalty: '处罚',
	      rumor: '传闻',
	      positive: '正向事实',
	      negative: '负向事实',
	      neutral: '中性事实',
	      mixed: '混合影响',
	      accepted: '已入主图',
	      pass: '契约通过',
	      trusted_main_graph: '主图可信',
	    })[stateType] || stateType || '待判定';
	  }

  function task4SeverityLabel(severity) {
    return ({ S4: '重大', S3: '高影响', S2: '中影响', S1: '低影响', S0: '观察' })[severity] || severity || '观察';
  }

  function task4VerificationLabel(status) {
    return ({
      not_human_gold: '非人工金标',
      agent_assisted_prototype: '辅助标注原型',
      product_demo_only: '仅产品演示',
      task4_core_complete_alert_prototype_only: '预警原型已完成',
      task4_completed_frozen_officially_evaluated: '正式评测完成',
      not_human_reviewed: '未人工复核',
      unknown: '待核验',
    })[status] || status || '待核验';
  }

  function task4LabelSourceLabel(source) {
    return ({
      agent_assisted_prototype: '模型辅助标注',
      manual_review: '人工复核',
      product_demo_only: '产品演示',
      official_target_mapping: '主体映射',
      schema_match: '事件模式匹配',
      title_semantic: '标题语义',
      body_semantic: '正文语义',
      all64_fallback: '综合兜底',
    })[source] || source || '待核验';
  }

  function task4ConfidenceLabel(value) {
    return ({ high: '高', medium: '中', low: '低' })[String(value)] || String(value || '—');
  }

  function task4PredictionStatusLabel(value) {
    return ({
      task4_core_complete_alert_prototype_only: '预警原型已完成',
      product_demo_only: '产品演示',
      task4_completed_frozen_officially_evaluated: '正式评测完成',
      complete: '已完成',
    })[value] || value || '预警原型';
  }

  function processStageTitle(stageKey) {
    return ({
      ingest: '进入队列',
      task1: '任务一',
      task2: '任务二',
      task3: '任务三',
      task4: '任务四',
    })[stageKey] || stageKey || '阶段';
  }

  function processStageBrief(stageKey) {
    return ({
      ingest: '事件从接入队列进入处理链路。',
      task1: '把相关信号归并成同一公开事件。',
      task2: '把公开来源和证据信号汇成可信判断。',
      task3: '锁定当前状态，便于持续追踪。',
      task4: '输出最终可直接阅读的提醒结果。',
    })[stageKey] || '';
  }

  function processStageResult(stageKey, entry = {}) {
    const task1 = entry.task1 || {};
    const task2 = entry.task2 || {};
    const task3 = entry.task3 || {};
    const task4 = entry.task4 || {};
    return ({
      ingest: `${entry.sourceCount || 0} 个来源`,
      task1: task1.verificationStatus || '已完成',
      task2: `${formatPercent(task2.score || 0)} · ${task2.level || '高可信'}`,
      task3: task3StateLabelLocal(task3.currentStateType || 'unknown'),
      task4: task4SeverityLabel(task4.sampleAlertCard?.severity || 'S0'),
    })[stageKey] || '—';
  }

  function processStageNote(stageKey, entry = {}) {
    const task1 = entry.task1 || {};
    const task2 = entry.task2 || {};
    const task3 = entry.task3 || {};
    const task4 = entry.task4 || {};
    return ({
      ingest: `${entry.title || '当前事件'} · ${entry.sourceCount || 0} 个来源，进入处理链路。`,
      task1: task1.summary || '相关信号已归并为同一公开事件。',
      task2: task2.summary || '系统结合公开来源和证据信号形成可信判断。',
      task3: task3.currentStateSummary || '系统根据事件脉络锁定当前状态。',
      task4: task4.summary || '系统把公开信号汇总为可直接阅读的提醒。',
    })[stageKey] || '';
  }

  function bandLabel(value) {
    if (value === null || value === undefined || value === '') return '未提供';
    const numeric = Number(value);
    return Number.isFinite(numeric) ? `${numeric.toFixed(0)} / 100` : '未提供';
  }

  function analysisContext(event) {
    return `<section class="panel analysis-context"><div><span class="eyebrow">当前分析对象</span><h2>${escapeHTML(event.title)}</h2><p>${escapeHTML(event.company)} · ${escapeHTML(event.code)} · ${escapeHTML(event.id)}</p></div><button class="secondary-button" data-action="select-event">切换事件</button></section>
      <nav class="analysis-steps" aria-label="事件分析步骤">${[['events','1','选择事件'],['credibility','2','可信评估'],['timeline','3','演化与影响'],['alerts','4','预警跟踪']].map(([route, step, label]) => `<button class="${state.route === route ? 'active' : ''}" data-nav="${route}" ${route === 'alerts' ? `data-event-filter="${escapeHTML(event.id)}"` : ''} ${state.route === route ? 'aria-current="step"' : ''}><span>${step}</span>${label}</button>`).join('')}</nav>`;
  }

  function selectionRequired() {
    $('#pageContainer').innerHTML = `<div class="page-content">${pageHeader(routes[state.route], '选择一个事件，查看其证据、演化过程和影响。')}<section class="panel empty-state"><strong>请先选择分析事件</strong><span>选定后，可信评估与事件脉络会围绕同一事件展开。</span><button class="primary-button" data-action="select-event">选择事件</button><button class="link-button" data-nav="events">浏览事件列表 →</button></section></div>`;
  }

  function routeFromLocation() {
    const [route, query = ''] = location.hash.replace(/^#\/?/, '').split('?');
    const params = new URLSearchParams(query);
    const selected = params.get('event') || '';
    state.selectedEvent = selected;
    state.alertFilters.eventId = route === 'alerts' && params.get('scope') !== 'all' ? selected : '';
    return navigate(route || 'dashboard', false);
  }

  async function perform(button, operation) {
    if (button?.disabled) return false;
    if (button) button.disabled = true;
    try { await operation(); return true; }
    catch (error) { showToast(error.message || '操作失败，请重试', true); return false; }
    finally { if (button) button.disabled = false; }
  }

  function friendlyStatusLabel(value) {
    const text = String(value || '').trim();
    if (!text) return '待核验';
    return ({
      unknown: '待核验',
      active: '进行中',
      done: '已完成',
      completed: '已完成',
      supported: '已确认',
      official_supported: '已确认',
      official_fulltext_supported: '已确认',
      reversed_or_contradicted: '存在冲突',
      pending: '待处理',
    })[text] || text.replace(/_/g, ' ');
  }

  function translateProcessScope(value) {
    return ({
      Severity: '严重度',
      'Alert Priority': '优先级',
      Validity: '有效性',
      'Impact Direction': '影响方向',
      risk_intensity: '风险强度',
      official_high_risk: '官方高风险',
      direction_gold: '影响方向',
      gate_coverage: '覆盖门控',
    })[value] || value;
  }

  function pageHeader(route, description, actions = '') {
    return `<header class="page-header">
      <div class="page-title-wrap"><span class="eyebrow">${route.eyebrow}</span><h1>${route.label}</h1><p>${description}</p></div>
      ${actions ? `<div class="page-actions">${actions}</div>` : ''}
    </header>`;
  }

  function panelHead(title, subtitle = '', actions = '') {
    return `<div class="panel-head"><div class="panel-title"><h2>${title}</h2>${subtitle ? `<p>${subtitle}</p>` : ''}</div>${actions ? `<div class="panel-actions">${actions}</div>` : ''}</div>`;
  }

  function processStageIndex(stageKey) {
    return PROCESS_STAGE_KEYS.indexOf(stageKey);
  }

  function stopProcessPlayback() {
    if (state.processDemo.timer) {
      window.clearTimeout(state.processDemo.timer);
      state.processDemo.timer = null;
    }
    state.processDemo.processingStage = -1;
    state.processDemo.processingEntryId = '';
    state.processDemo.runToken += 1;
  }

  function activeProcessData() {
    return state.processDemo.data;
  }

  function currentProcessEntry() {
    const data = activeProcessData();
    if (!data?.entries?.length || state.processDemo.activeIndex < 0) return null;
    return data.entries[state.processDemo.activeIndex] || null;
  }

  function currentProcessStage() {
    const data = activeProcessData();
    if (!data?.stageOrder?.length || state.processDemo.stageIndex < 0) return null;
    return data.stageOrder[state.processDemo.stageIndex] || null;
  }

  function setProcessSelection(entryIndex, stageIndex = -1) {
    const data = activeProcessData();
    if (!data?.entries?.length) return;
    const safeEntry = Number.isFinite(entryIndex) && entryIndex >= 0 && entryIndex < data.entries.length ? entryIndex : -1;
    const entryChanged = safeEntry !== state.processDemo.activeIndex;
    if (entryChanged) {
      stopProcessPlayback();
      state.processDemo.completedStage = -1;
      state.processDemo.stageIndex = safeEntry < 0 ? -1 : 0;
    }
    state.processDemo.activeIndex = safeEntry;
    const safeStage = Number.isFinite(stageIndex) && stageIndex >= 0 && data.stageOrder?.[stageIndex] ? stageIndex : state.processDemo.stageIndex;
    if (safeEntry < 0) {
      state.processDemo.stageIndex = -1;
    } else if (safeStage <= state.processDemo.completedStage || safeStage === state.processDemo.completedStage + 1) {
      state.processDemo.stageIndex = safeStage;
    }
    renderIntegrationView();
  }

  function moveProcessPlayback(delta = 1) {
    const data = activeProcessData();
    if (!data?.entries?.length || state.route !== 'integration' || state.processDemo.activeIndex < 0) return;
    const entryIndex = ((state.processDemo.activeIndex + delta) % data.entries.length + data.entries.length) % data.entries.length;
    state.processDemo.activeIndex = entryIndex;
    state.processDemo.stageIndex = -1;
    renderIntegrationView();
  }

  function startProcessPlayback() {
    stopProcessPlayback();
    const data = activeProcessData();
    if (state.route !== 'integration' || !data?.entries?.length) return;
    const interval = Math.max(2200, data.playbackIntervalMs || 3200);
    state.processDemo.timer = window.setInterval(() => moveProcessPlayback(1), interval);
  }

  function processStageState(index) {
    if (state.processDemo.processingStage === index) return 'processing';
    if (index <= state.processDemo.completedStage) return 'completed';
    if (index === state.processDemo.completedStage + 1) return 'ready';
    return 'locked';
  }

  function processStageStateLabel(status) {
    return ({ completed: '已完成', processing: '处理中', ready: '可开始', locked: '待解锁' })[status] || '待处理';
  }

  function processStageReadyMarkup(entry, stage, stageIndex) {
    const isIngest = stage.key === 'ingest';
    const actionLabel = isIngest ? '开始接收事件' : `开始处理：${stage.label}`;
    const description = isIngest
      ? '系统将读取该事件的公开来源，并把它放入处理队列。'
      : processStageBrief(stage.key);
    return `<div class="process-simulation-ready">
      <div class="process-simulation-icon" style="--stage-color:${stage.color || COLORS.blue}">${escapeHTML(stage.icon || '○')}</div>
      <span class="eyebrow">${escapeHTML(isIngest ? '等待接入' : '等待执行')}</span>
      <h3>${escapeHTML(stage.label)}</h3>
      <p>${escapeHTML(description)}</p>
      ${isIngest ? `<div class="process-intake-preview">${processDetailRows([
        ['事件标题', entry.title || '—'],
        ['公开来源', `${entry.sourceCount || 0} 个`],
        ['接入状态', '待开始'],
      ])}</div>` : ''}
      <button class="primary-button process-start-button" data-action="process-start" data-stage-index="${stageIndex}">${escapeHTML(actionLabel)}</button>
      
    </div>`;
  }

  function processStageProcessingMarkup(stage) {
    const messages = ({
      ingest: '正在读取公开来源并建立事件输入…',
      task1: '正在检索相关信号并归并事件成员…',
      task2: '正在比对公开证据并形成可信判断…',
      task3: '正在串联事件脉络并判断当前状态…',
      task4: '正在汇总影响信号并生成预警输出…',
    });
    return `<div class="process-simulation-processing">
      <div class="process-loading-ring" style="--stage-color:${stage.color || COLORS.blue}"></div>
      <span class="eyebrow">正在处理</span>
      <h3>${escapeHTML(stage.label)}</h3>
      <p>${escapeHTML(messages[stage.key] || '系统正在处理当前阶段…')}</p>
      <div class="process-progress"><span></span></div>
      <div class="process-live-steps"><span class="done">✓ 校验事件上下文</span><span class="active">● 执行当前任务</span><span>○ 等待输出结果</span></div>
      
    </div>`;
  }

  function processStagePreview(index, entry = {}) {
    const stage = activeProcessData()?.stageOrder?.[index];
    if (!stage || !entry) return '待选择';
    const status = processStageState(index);
    return status === 'completed' ? processStageResult(stage.key, entry) : processStageStateLabel(status);
  }

  function startProcessStage(stageIndex) {
    const data = activeProcessData();
    const entry = currentProcessEntry();
    if (!data?.stageOrder?.[stageIndex] || !entry || state.processDemo.processingStage >= 0) return;
    if (stageIndex !== state.processDemo.completedStage + 1) return;
    const stage = data.stageOrder[stageIndex];
    stopProcessPlayback();
    state.processDemo.stageIndex = stageIndex;
    state.processDemo.processingStage = stageIndex;
    state.processDemo.processingEntryId = entry.id;
    const token = state.processDemo.runToken;
    renderIntegrationView();
    const duration = ({ ingest: 1200, task1: 1900, task2: 2100, task3: 2200, task4: 1800 })[stage.key] || 1800;
    state.processDemo.timer = window.setTimeout(() => {
      if (token !== state.processDemo.runToken || state.processDemo.processingEntryId !== entry.id) return;
      state.processDemo.timer = null;
      state.processDemo.processingStage = -1;
      state.processDemo.processingEntryId = '';
      state.processDemo.completedStage = stageIndex;
      state.processDemo.stageIndex = stageIndex;
      renderIntegrationView();
    }, duration);
  }

  function processDetailRows(rows) {
    return `<div class="process-detail-grid">${rows.map(([label, value, tone = '']) => `<div class="process-detail"><span>${escapeHTML(label)}</span><b class="${tone}">${escapeHTML(value)}</b></div>`).join('')}</div>`;
  }

  function processStageMarkup(entry, stage) {
    const task1 = entry.task1 || {};
    const task2 = entry.task2 || {};
    const task3 = entry.task3 || {};
    const task4 = entry.task4 || {};
    if (!stage) return '<div class="empty-state"><strong>暂无流程数据</strong></div>';
    if (stage.key === 'ingest') {
      return `
        <div class="process-card-block">
          <span class="eyebrow">事件进入</span>
          <h3>${escapeHTML(entry.title)}</h3>
          <p>公开来源 ${escapeHTML(String(entry.sourceCount || 0))} 个 · 已进入处理链路</p>
          ${processDetailRows([
            ['公开来源', `${entry.sourceCount || 0} 个`],
            ['影响级别', levelLabel(entry.level || 'low')],
            ['情绪方向', sentimentLabel(entry.sentiment || 'neutral')],
            ['当前状态', '进入队列'],
          ])}
          <div class="process-tag-row">${[entry.company, entry.code, entry.source, levelLabel(entry.level || 'low')].filter(Boolean).map((item) => `<span class="mini-tag">${escapeHTML(item)}</span>`).join('')}</div>
          <p class="process-note">公开事件从接入队列进入处理链路。</p>
        </div>`;
    }
    if (stage.key === 'task1') {
      const sources = (task1.sourceList || []).slice(0, 4);
      const previews = (task1.preview || []).slice(0, 3);
      return `
        <div class="process-card-block">
          <span class="eyebrow">任务一 · 事件归并</span>
          <h3>${escapeHTML(task1.eventName || entry.title)}</h3>
          <p>${escapeHTML(task1.summary || '系统将相关信号归并为同一公开事件。')}</p>
          ${processDetailRows([
            ['归并事件', task1.eventName || entry.title],
            ['关联主体', task1.targetName || entry.company],
            ['来源数量', `${task1.supportingSources || entry.sourceCount || 0} 个`],
            ['成员新闻', `${task1.memberCount || 0} 条`],
            ['归并结果', task1.verificationStatus || '已完成'],
          ])}
          <div class="process-tag-row">${[task1.verificationStatus, task1.targetName, task1.eventName, task1.targetCode].filter(Boolean).map((item) => `<span class="mini-tag">${escapeHTML(item)}</span>`).join('')}</div>
          ${sources.length ? `<div class="process-evidence-list">${sources.map((item) => `<div class="process-evidence-item"><strong>${escapeHTML(item)}</strong><span>归并来源</span><p>已纳入同一事件簇</p></div>`).join('')}</div>` : ''}
          ${previews.length ? `<div class="process-evidence-list" style="margin-top:8px">${previews.map((item, index) => `<div class="process-evidence-item"><strong>线索 ${index + 1}</strong><span>${escapeHTML(task1.eventName || entry.title)}</span><p>${escapeHTML(item)}</p></div>`).join('')}</div>` : ''}
          <p class="process-note">这一步把分散信息收拢成同一事件，便于后续判断。</p>
        </div>`;
    }
    if (stage.key === 'task2') {
      const evidenceItems = (task2.evidenceItems || []).slice(0, 3);
      return `
        <div class="process-card-block">
          <span class="eyebrow">任务二 · 公开判断</span>
          <h3>${escapeHTML(task2.level || '可信判断')}</h3>
          <p>${escapeHTML(task2.supportingReasons || task2.reason || '系统结合公开来源和证据信号形成可信判断。')}</p>
          ${processDetailRows([
            ['综合评分', formatPercent(task2.score || 0)],
            ['结果等级', task2.level || '高可信'],
            ['证据句数', `${task2.evidenceSentenceCount || 0} 条`],
            ['官方支撑', `${task2.officialSupportCount || 0} 条`],
          ])}
          <div class="process-tag-row">${[bandLabel(task2.score || 0), task2.verificationStatus, task2.level].filter(Boolean).map((item) => `<span class="mini-tag">${escapeHTML(item)}</span>`).join('')}</div>
          ${evidenceItems.length ? `<div class="process-evidence-list">${evidenceItems.map((item) => `<div class="process-evidence-item"><strong>${escapeHTML(item.source)}</strong><span>${escapeHTML(item.status || item.type || '')}</span><p>${escapeHTML(item.text || item.title || '')}</p></div>`).join('')}</div>` : ''}
          <p class="process-note">这里看的是公开来源与证据信号的综合判断。</p>
        </div>`;
    }
    if (stage.key === 'task3') {
      const trace = (entry.trace || []).slice(0, 5);
      return `
        <div class="process-card-block">
          <span class="eyebrow">任务三 · 脉络追踪</span>
          <h3>${escapeHTML(task3StateLabelLocal(task3.currentStateType || 'unknown'))}</h3>
          <p>${escapeHTML(task3.currentStateSummary || task3.summary || '系统根据事件脉络锁定当前状态。')}</p>
          ${processDetailRows([
            ['当前状态', task3StateLabelLocal(task3.currentStateType || 'unknown')],
            ['信任度', formatPercent((task3.trustScore || 0) * 100)],
            ['状态置信度', task3.stateConfidence == null ? '未提供' : bandLabel(task3.stateConfidence * 100)],
            ['核验结果', friendlyStatusLabel(task3.stateVerificationStatus)],
          ])}
          <div class="process-tag-row">${[task3StateLabelLocal(task3.currentStateType || 'unknown'), friendlyStatusLabel(task3.stateVerificationStatus), bandLabel((task3.trustScore || 0) * 100)].filter(Boolean).map((item) => `<span class="mini-tag">${escapeHTML(item)}</span>`).join('')}</div>
          ${trace.length ? `<div class="process-path"><span class="eyebrow">流程轨迹</span><div class="process-path-list">${trace.map((item, index) => `<span class="${index === state.processDemo.stageIndex ? 'active' : ''}"><b>${index + 1}</b>${escapeHTML(item.label || item.phase || `步骤${index + 1}`)} · ${escapeHTML(item.note || item.state || '')}</span>`).join('')}</div></div>` : ''}
          <p class="process-note">这一层只保留公开的当前状态。</p>
        </div>`;
    }
    const alertCard = task4.sampleAlertCard || {};
    const severity = alertCard.severity || 'S0';
    return `
        <div class="process-card-block">
        <span class="eyebrow">任务四 · 预警输出</span>
        <h3>${escapeHTML(alertCard.title || task4.candidate?.predictedEventName || '重点提醒')}</h3>
        <p>${escapeHTML(task4.summary || alertCard.reason || '系统把公开信号汇总为可直接阅读的提醒。')}</p>
        ${processDetailRows([
          ['预警主题', alertCard.title || '—'],
          ['影响范围', alertCard.impactScope || '—'],
          ['严重度', task4SeverityLabel(severity)],
          ['提醒等级', alertCard.alertPriority || '—'],
        ])}
        <div class="process-tag-row">${[task4SeverityLabel(severity), alertCard.alertPriority, alertCard.impactScope, task4PredictionStatusLabel(task4.formalPrediction?.status || alertCard.verificationStatus || '')].filter(Boolean).map((item) => `<span class="mini-tag">${escapeHTML(item)}</span>`).join('')}</div>
        ${renderIntegrationAlertCard({
          title: alertCard.title || task4.candidate?.predictedEventName || '',
          eventName: task4.candidate?.predictedTargetName || entry.company || '',
          scope: alertCard.impactScope || '',
          importance: `${task4SeverityLabel(severity)} / ${alertCard.alertPriority || ''}`.trim(),
          level: ({ S4: 'critical', S3: 'high', S2: 'medium', S1: 'low', S0: 'low' })[severity] || 'medium',
          color: ({ S4: '#ff5d73', S3: '#ffb648', S2: '#4aa8ff', S1: '#9677ff', S0: '#4ce3ef' })[severity] || '#ffb648',
          icon: ({ S4: '!', S3: '↗', S2: '◇', S1: '•', S0: '·' })[severity] || '◇',
          status: task4PredictionStatusLabel(task4.formalPrediction?.status || alertCard.verificationStatus || '已完成'),
          severity,
          priority: alertCard.alertPriority || '',
          urgency: alertCard.urgency || '',
          duration: alertCard.duration || '',
          confidence: alertCard.prototypeConfidence || '',
          verificationStatus: alertCard.verificationStatus || '',
          labelSource: alertCard.labelSource || '',
          reason: task4.summary || alertCard.reason || '',
          quote: alertCard.evidenceQuote || '',
          directEvidence: alertCard.directEvidence || '',
        })}
        <p class="process-note">这里展示的是最终可以直接给用户看的提醒结果。</p>
      </div>`;
  }

  function renderIntegrationViewLegacy() {
    const data = activeProcessData();
    if (!data?.entries?.length) return;
    const queueItems = data.queue || [];
    const stageCards = data.stageOrder || [];
    const taskStageCards = stageCards.filter((item) => item.key !== 'ingest');
    const selectedQueueItem = state.processDemo.activeIndex >= 0 ? queueItems[state.processDemo.activeIndex] || null : null;
    const selectedEntry = selectedQueueItem ? (data.entries || []).find((item) => item.id === selectedQueueItem.id) || null : null;
    const selectedStage = currentProcessStage();
    const selectedStageIndex = state.processDemo.stageIndex >= 0 ? state.processDemo.stageIndex : -1;
    const selectedStageLabel = selectedStage?.label || '待选择任务';
    const selectedStageResult = selectedStage && selectedEntry ? processStageResult(selectedStage.key, selectedEntry) : '未加载';
    const sourceBase = Math.max(1, Number(selectedQueueItem?.sourceCount || selectedEntry?.sourceCount || 0));
    const topCards = [
      { label: '事件池', value: formatNumber(queueItems.length), delta: '待选事件', note: '从下拉框选择后进入队列', icon: '样', color: COLORS.primary, spark: [queueItems.length, queueItems.length + 1, queueItems.length + 2, queueItems.length + 3, queueItems.length + 2, queueItems.length + 4, queueItems.length + 5, queueItems.length + 6] },
      { label: '当前事件', value: selectedQueueItem?.company || '待选择事件', delta: selectedQueueItem?.code || '等待选择', note: selectedQueueItem ? '已选为回放对象' : '只有选中后才显示队列内容', icon: '事', color: COLORS.blue, spark: [sourceBase, sourceBase + 1, sourceBase + 2, sourceBase + 1, sourceBase + 3, sourceBase + 4, sourceBase + 3, sourceBase + 5] },
      { label: '当前步骤', value: selectedStageLabel, delta: selectedQueueItem ? ('已载入') : '等待选择', note: selectedQueueItem ? '选择任务查看已有结果' : '先选事件，再选任务', icon: '步', color: COLORS.purple, spark: [selectedStageIndex + 1, selectedStageIndex + 2, selectedStageIndex + 3, selectedStageIndex + 2, selectedStageIndex + 4, selectedStageIndex + 5, selectedStageIndex + 4, selectedStageIndex + 6] },
      { label: '当前结果', value: selectedStageResult, delta: selectedStage ? selectedStage.label : '未加载', note: '未选择的步骤保持收起', icon: '结', color: COLORS.warning, spark: [12, 15, 18, 20, 22, 24, 25, 27] },
    ];
    const route = routes.integration;
    $('#breadcrumb').innerHTML = `<strong>${route.label}</strong>`;
    $('#pageContainer').innerHTML = `<div class="page-content process-page">
      ${pageHeader(route, '选择事件，再查看各任务的已有输出。本页为结果回放，切换阶段不会触发模型重新推理。', `<button class="ghost-button" data-action="process-reset">↺ 清空选择</button><button class="ghost-button" data-nav="dashboard">⌂ 返回概览</button>`)}
      <section class="kpi-grid">
        ${topCards.map((item, index) => `<article class="panel kpi-card" style="--card-color:${item.color}">
          <div class="kpi-top"><span>${item.label}</span><i class="kpi-icon">${item.icon}</i></div>
          <div class="kpi-value">${item.value}</div>
          <div class="kpi-bottom"><span><b class="delta ${item.delta.startsWith('-') ? 'negative' : ''}">${item.delta}</b> ${item.note}</span></div>
        </article>`).join('')}
      </section>
      <section class="panel process-hero">
        <div class="process-hero-head">
          <div class="process-hero-copy">
            <span class="eyebrow">任务链路 · 事件进入</span>
            <h2>${escapeHTML(selectedQueueItem?.title || '请选择事件进入队列')}</h2>
            <p>${escapeHTML(selectedQueueItem ? '当前事件已选中，选择任务查看已有结果。' : '先选择事件，再按需展开各任务的已有结果。')}</p>
          </div>
          <div class="process-hero-stats">
            <div><span>当前事件</span><b>${escapeHTML(selectedQueueItem?.title || '待选择')}</b></div>
            <div><span>公开来源</span><b>${escapeHTML(selectedQueueItem ? `${String(selectedQueueItem.sourceCount || 0)} 个` : '—')}</b></div>
            <div><span>当前步骤</span><b>${escapeHTML(selectedStageLabel)}</b></div>
            <div><span>当前结果</span><b>${escapeHTML(selectedStageResult)}</b></div>
          </div>
        </div>
      </section>
      <section class="process-layout">
        <aside class="panel process-event-rail">
          ${panelHead('回放事件列表', '通过下拉框选择事件，未选中前不显示队列内容', `<select class="select-control process-select" id="processEventSelect" aria-label="选择事件"><option value="">请选择事件</option>${queueItems.map((item, index) => `<option value="${index}" ${index === state.processDemo.activeIndex ? 'selected' : ''}>${escapeHTML(item.title)} · ${escapeHTML(item.company)} · ${escapeHTML(String(item.sourceCount || 0))}源</option>`).join('')}</select>`)}
          <div class="process-queue-list">
            ${selectedQueueItem ? `<article class="process-queue-item active">
              <span class="process-queue-index">${escapeHTML(String(selectedQueueItem.rank || 1))}</span>
              <div class="process-queue-body">
                <strong>${escapeHTML(selectedQueueItem.title)}</strong>
                <span>${escapeHTML(selectedQueueItem.company)} · ${escapeHTML(selectedQueueItem.code)}</span>
              </div>
              <div class="process-queue-meta">
                <b>${escapeHTML('已载入')}</b>
                <small>${escapeHTML(String(selectedQueueItem.sourceCount || 0))} 个来源</small>
              </div>
            </article>` : `<div class="empty-state process-queue-empty"><strong>尚未选择回放事件</strong><span>请先从上方下拉框选择一个事件。</span></div>`}
          </div>
        </aside>
        <div class="process-main">
          <section class="panel process-step-panel">
            ${panelHead('处理步骤', '选择任务查看已有结果', `<select class="select-control process-select" id="processStageSelect" aria-label="选择任务" ${selectedQueueItem ? '' : 'disabled'}><option value="">请选择任务</option>${taskStageCards.map((item) => {
              const stageIndex = stageCards.findIndex((stage) => stage.key === item.key);
              return `<option value="${stageIndex}" ${selectedStageIndex === stageIndex ? 'selected' : ''}>${escapeHTML(item.label)}</option>`;
            }).join('')}</select>`)}
            <div class="process-step-list">
              ${taskStageCards.map((item) => {
                const index = stageCards.findIndex((stage) => stage.key === item.key);
                return `<button class="process-step-chip ${processStageState(index)} ${index === selectedStageIndex ? 'active' : ''}" data-process-stage="${index}" style="--stage-color:${item.color}">
                <span class="process-step-icon">${escapeHTML(item.icon || String(index + 1))}</span>
                <span class="process-step-copy">
                  <strong>${escapeHTML(item.label)}</strong>
                  <small>${escapeHTML(processStageBrief(item.key))}</small>
                </span>
                <span class="process-step-state">${escapeHTML(processStagePreview(index, selectedEntry))}</span>
              </button>`;
              }).join('')}
            </div>
          </section>
          <section class="panel process-stage-panel">
            ${panelHead(selectedStage?.label || '阶段结果', `${selectedEntry ? `${selectedEntry.title} · ${selectedEntry.company}` : '选择事件后再加载任务'}`, `<span class="pill" style="--pill-color:${selectedStage?.color || '#2ee6a6'}">${selectedStage ? '当前阶段' : '等待选择'}</span>`)}
            <div class="process-stage-main">
              ${selectedQueueItem && selectedStage ? processStageMarkup(selectedEntry, selectedStage) : '<div class="empty-state process-stage-empty"><strong>请选择任务后加载对应信息</strong><span>选择阶段仅展开已有结果，不改变实际处理状态。</span></div>'}
            </div>
          </section>
          <div class="process-summary-row">
            <section class="panel process-summary-panel">
              ${panelHead('当前事件概览', '选择事件后可逐步查看任务结果', `<span class="pill" style="--pill-color:#2f75f6">已载入数据</span>`)}
              <div class="process-summary-grid">
                <div class="process-summary-item"><span>事件标题</span><b>${escapeHTML(selectedQueueItem?.title || '待选择事件')}</b><small>${escapeHTML(selectedQueueItem ? `${selectedQueueItem.company} · ${selectedQueueItem.code}` : '请先从下拉框选择事件')}</small></div>
                <div class="process-summary-item"><span>来源信息</span><b>${escapeHTML(selectedQueueItem?.source || '—')}</b><small>${escapeHTML(selectedQueueItem ? `${String(selectedQueueItem.sourceCount || 0)} 个来源` : '选择后才显示')}</small></div>
                <div class="process-summary-item"><span>队列状态</span><b>${escapeHTML(selectedQueueItem?.queueLabel || '待选择')}</b><small>${escapeHTML(selectedQueueItem ? `排名 ${String(selectedQueueItem.rank || 1)}` : '未进入队列')}</small></div>
                <div class="process-summary-item"><span>当前步骤</span><b>${escapeHTML(selectedStageLabel)}</b><small>${escapeHTML(selectedStage ? processStageBrief(selectedStage.key) : '选择任务查看已有结果')}</small></div>
              </div>
            </section>
            <section class="panel process-summary-panel">
              ${panelHead('阶段加载状态', '未选择的任务保持待选择状态', `<span class="pill" style="--pill-color:#ffb648">按需展开</span>`)}
              <div class="process-summary-grid">
                ${taskStageCards.map((item, index) => {
                  const demoIndex = stageCards.findIndex((stage) => stage.key === item.key);
                  return `<div class="process-summary-item ${selectedStageIndex === demoIndex ? 'active' : ''}" data-process-stage="${demoIndex}">
                    <span>${escapeHTML(item.label)}</span>
                    <b>${escapeHTML(processStagePreview(demoIndex, selectedEntry))}</b>
                    <small>${escapeHTML(processStageBrief(item.key))}</small>
                  </div>`;
                }).join('')}
              </div>
            </section>
          </div>
        </div>
      </section>
    </div>`;

  }

  function renderNavigation() {
    const sections = [...new Set(Object.values(routes).map((route) => route.section))];
    $('#primaryNav').innerHTML = sections.map((section) => `
      <div class="nav-section-title">${section}</div>
      ${Object.entries(routes).filter(([, route]) => route.section === section).map(([key, route]) => `
        <button class="nav-item ${state.route === key ? 'active' : ''}" data-nav="${key}" ${state.route === key ? 'aria-current="page"' : ''}>
          <span class="nav-icon">${route.icon}</span><span>${route.label}</span>${route.badge ? `<span class="nav-badge">${route.badge}</span>` : ''}
        </button>`).join('')}`).join('');
  }

  function setBreadcrumb() {
    const route = routes[state.route];
    $('#breadcrumb').innerHTML = `<strong>${route.label}</strong>`;
  }

  function setLoading() {
    $('#pageContainer').innerHTML = '<div class="page-loading"><div class="loading-ring"></div><span>正在加载分析数据…</span></div>';
  }

  function navigationIsStale(route, navigationId) {
    return state.route !== route || (navigationId !== undefined && navigationId !== state.navigationId);
  }

  async function navigate(route, updateHash = true) {
    if (!routes[route]) route = 'dashboard';
    const navigationId = ++state.navigationId;
    state.route = route;
    const contextId = route === 'alerts' && state.alertFilters.eventId ? state.alertFilters.eventId : state.selectedEvent;
    const hash = `#/${route}${contextId ? `?event=${encodeURIComponent(contextId)}${route === 'alerts' && !state.alertFilters.eventId ? '&scope=all' : ''}` : ''}`;
    if (updateHash && location.hash !== hash) history.pushState(null, '', hash);
    stopProcessPlayback();
    renderNavigation(); setBreadcrumb(); setLoading(); closePanels(); closeDrawer(); closeSearch();
    const renderers = { dashboard: renderDashboard, events: renderEvents, credibility: renderCredibility, timeline: renderTimeline, graph: renderGraph, alerts: renderAlerts, integration: renderIntegration };
    try {
      await renderers[route](navigationId);
      if (navigationIsStale(route, navigationId)) return;
      $('#pageContainer').scrollTop = 0;
      $('#pageContainer').focus({ preventScroll: true });
      return true;
    } catch (error) {
      console.error(error);
      if (navigationIsStale(route, navigationId)) return;
      $('#pageContainer').innerHTML = '<div class="empty-state"><span style="font-size:32px">!</span><strong>页面加载失败</strong><span>' + escapeHTML(error.message || '请稍后重试') + '</span><button class="link-button" data-action="select-event">重新选择事件</button><button class="secondary-button" style="margin-top:14px" data-nav="' + route + '">重新加载</button></div>';
    }
  }

  async function renderDashboard(navigationId) {
    const [data, events, alerts] = await Promise.all([API.getDashboard(), API.getEvents(), API.getAlerts()]);
    if (navigationIsStale('dashboard', navigationId)) return;
    const route = routes.dashboard;
    const highImpactCount = events.filter((event) => ['high', 'critical'].includes(event.level)).length;
    const criticalCount = alerts.filter((alert) => alert.level === 'critical').length;
    const companyCount = new Set(events.map((event) => event.company)).size;
    const topTheme = data.composition?.[0]?.name || '公开事件';
    const updates = data.updates || [];
    const topCards = [
      { label: '最新态势', value: highImpactCount ? '活跃' : '稳定', delta: `高影响 ${formatNumber(highImpactCount)}`, note: '当前公开事件样本', icon: '势', color: COLORS.primary, spark: data.trend.total.slice(-8) },
      { label: '重点预警', value: formatNumber(alerts.length), delta: '优先关注', note: `重大 ${formatNumber(criticalCount)} 条`, icon: '警', color: COLORS.warning, spark: data.trend.high.slice(-8) },
      { label: '覆盖主体', value: formatNumber(companyCount), delta: `${formatNumber(events.length)} 条样本`, note: '不同公司主体已聚合', icon: '主', color: COLORS.blue, spark: data.trend.total.slice(-8).map((value, index) => Math.max(1, value - index * 2)) },
      { label: '主题分布', value: topTheme, delta: '最近热点', note: '按公开事件聚合', icon: '类', color: COLORS.purple, spark: data.composition.map((item) => Math.round(item.value)).concat([12, 15, 18, 14]) .slice(0, 8) },
    ];
    $('#pageContainer').innerHTML = `<div class="page-content">
      ${pageHeader(route, '聚合最新公开事件，突出重点信号、预警提醒与可见趋势。', `
        <button class="ghost-button" data-nav="integration">◈ 任务链路</button><button class="ghost-button" data-action="refresh">↻ 刷新数据</button><button class="primary-button" data-action="create-watch">＋ 保存关注</button>`)}
      <section class="kpi-grid">
        ${topCards.map((item, index) => `<article class="panel kpi-card" style="--card-color:${item.color}">
          <div class="kpi-top"><span>${item.label}</span><i class="kpi-icon">${item.icon}</i></div>
          <div class="kpi-value">${item.value}</div>
          <div class="kpi-bottom"><span><b class="delta ${item.delta.startsWith('-') ? 'negative' : ''}">${item.delta}</b> ${item.note}</span></div>
        </article>`).join('')}
      </section>
      <section class="dashboard-grid">
        <article class="panel chart-card">
          ${panelHead('事件态势趋势', '按样本日期统计事件数量与高影响事件', `<div class="chart-legend"><span><i style="--legend-color:#2f75f6"></i>全部事件</span><span><i style="--legend-color:#ff9f2e"></i>高影响</span></div><span class="mini-tag">最近 ${data.trend.labels.length} 个样本日期</span>`)}
          <div class="panel-body"><div class="chart-wrap"><canvas id="trendChart"></canvas></div></div>
        </article>
        <article class="panel event-composition">
          ${panelHead('事件类型构成', '展示当前事件的分布情况')}
          <div class="panel-body donut-layout">
            <div class="donut-wrap"><canvas id="compositionChart"></canvas><div class="donut-center"><strong>事件分布</strong><span>多源汇总</span></div></div>
            <div class="composition-list">${data.composition.map((item) => `<div class="composition-item" style="--color:${item.color}"><span><i></i>${item.name}</span><b>${item.value}%</b><div class="progress"><i style="width:${item.value}%"></i></div></div>`).join('')}</div>
          </div>
        </article>
      </section>
      <section class="dashboard-lower">
        <article class="panel event-feed">
          ${panelHead('重点事件流', '按影响度与公开信号排序', `<button class="link-button" data-nav="events">进入事件中心 →</button>`)}
          <div>${[...events].sort((a, b) => b.impact - a.impact).slice(0, 6).map(eventFeedRow).join('') || '<div class="empty-state">暂无事件数据</div>'}</div>
        </article>
        <article class="panel pipeline-panel">
          ${panelHead('近期更新', '最近载入的项目结果', `<span class="pill" style="--pill-color:#2ee6a6">最新同步</span>`)}
          <div class="log-list">${updates.map((item) => `<div class="log-row"><time>${escapeHTML(item.time || '最新')}</time><span class="log-level success">${escapeHTML(item.title || '更新')}</span><p>${escapeHTML(item.note || '')}</p><span>${escapeHTML(item.status || '可读')}</span></div>`).join('')}</div>
        </article>
      </section>
    </div>`;
    Charts.line($('#trendChart'), data.trend);
    Charts.donut($('#compositionChart'), data.composition);
  }

  function eventFeedRow(event) {
    return `<div class="feed-row" data-event-id="${event.id}" role="button" tabindex="0">
      <span class="feed-company" style="--logo-color:${event.level === 'critical' ? '#ff5d73' : event.level === 'high' ? '#ffb648' : '#4aa8ff'}">${event.logo}</span>
      <span class="feed-content"><h4>${escapeHTML(event.title)}</h4><span class="feed-meta"><b>${event.company} · ${event.code}</b><span>${event.time}</span><span class="mini-tag">${event.type}</span><span class="status-pill status-${event.level}">${levelLabel(event.level)}</span></span></span>
      <span class="feed-score"><strong>${levelLabel(event.level)}</strong><span>影响级别</span></span>
    </div>`;
  }

  async function renderEvents(navigationId = state.navigationId) {
    if (state.route !== 'events') return;
    const requestId = ++state.eventRequest;
    const filters = { ...state.eventFilters };
    const events = await API.getEvents(filters);
    if (navigationIsStale('events', navigationId) || requestId !== state.eventRequest || JSON.stringify(filters) !== JSON.stringify(state.eventFilters)) return;
    const pageCount = Math.max(1, Math.ceil(events.length / 25));
    state.eventPage = Math.max(1, Math.min(state.eventPage, pageCount));
    const pageStart = (state.eventPage - 1) * 25;
    const visibleEvents = events.slice(pageStart, pageStart + 25);
    const input = $('#eventSearch');
    const focused = document.activeElement === input;
    const selection = focused ? [input.selectionStart, input.selectionEnd] : null;
    const route = routes.events;
    $('#pageContainer').innerHTML = `<div class="page-content">
      ${pageHeader(route, '统一管理公司级、行业级与产业链事件，支持条件筛选、逐页浏览和明细追溯。', `<button class="ghost-button" data-action="export">⇩ 导出事件</button><button class="primary-button" data-action="create-watch">＋ 保存关注</button>`)}
      <section class="panel toolbar">
        <div class="filter-group">
          <label class="search-input"><span>⌕</span><input class="input-control" id="eventSearch" value="${escapeHTML(state.eventFilters.query)}" placeholder="搜索事件标题、公司或代码"></label>
          <select class="select-control" id="levelFilter" aria-label="影响级别"><option value="all">全部影响级别</option><option value="critical" ${state.eventFilters.level === 'critical' ? 'selected' : ''}>重大</option><option value="high" ${state.eventFilters.level === 'high' ? 'selected' : ''}>高影响</option><option value="medium" ${state.eventFilters.level === 'medium' ? 'selected' : ''}>中影响</option><option value="low" ${state.eventFilters.level === 'low' ? 'selected' : ''}>低影响</option></select>
          <select class="select-control" id="sentimentFilter" aria-label="情感方向"><option value="all">全部情感</option><option value="positive" ${state.eventFilters.sentiment === 'positive' ? 'selected' : ''}>正面</option><option value="negative" ${state.eventFilters.sentiment === 'negative' ? 'selected' : ''}>负面</option><option value="neutral" ${state.eventFilters.sentiment === 'neutral' ? 'selected' : ''}>中性</option></select>
        </div>
        <div class="filter-group"><button class="link-button" data-action="reset-filters">重置筛选</button><span class="result-count">共 ${events.length} 条结果</span></div>
      </section>
      <section class="panel table-panel" aria-live="polite">
        ${events.length ? `<table class="data-table"><thead><tr><th>事件信息</th><th>关联主体</th><th>事件类型</th><th>情感</th><th>可信度</th><th>影响级别</th><th>状态</th><th>操作</th></tr></thead><tbody>
          ${visibleEvents.map((event) => `<tr data-event-id="${event.id}"><td class="event-title-cell"><strong>${escapeHTML(event.title)}</strong><span>${event.time} · ${event.source} · ${event.sourceCount} 个来源</span></td>
            <td><div class="company-cell"><i>${event.logo}</i><span><strong>${event.company}</strong><span>${event.code}</span></span></div></td>
            <td><span class="mini-tag">${event.type}</span></td><td><span style="color:${event.sentiment === 'positive' ? 'var(--primary)' : event.sentiment === 'negative' ? 'var(--danger)' : 'var(--muted)'}">${sentimentLabel(event.sentiment)}</span></td>
            <td><span class="credibility-value">${bandLabel(event.credibility)}</span><small class="cell-note">${escapeHTML(event.credibilityLevel || "")}</small></td>
            <td><span class="status-pill status-${event.level}">${levelLabel(event.level)}</span></td><td>${event.status}</td><td><div class="row-actions"><button class="analyze-link" data-nav="credibility" data-analyze-id="${escapeHTML(event.id)}">分析</button><button title="查看详情" data-event-id="${event.id}">›</button></div></td></tr>`).join('')}
        </tbody></table><div class="pagination"><span>显示 ${pageStart + 1}–${Math.min(pageStart + 25, events.length)}，共 ${events.length} 条</span><div class="pagination-buttons"><button aria-label="上一页" data-action="event-prev" ${state.eventPage === 1 ? 'disabled' : ''}>‹</button><span>${state.eventPage} / ${pageCount}</span><button aria-label="下一页" data-action="event-next" ${state.eventPage === pageCount ? 'disabled' : ''}>›</button></div></div>` : `<div class="empty-state"><span style="font-size:32px">⌕</span><strong>未找到匹配事件</strong><span>请调整搜索关键词或筛选条件</span></div>`}
      </section>
    </div>`;
    if (focused && $('#eventSearch')) { $('#eventSearch').focus({ preventScroll: true }); try { $('#eventSearch').setSelectionRange(...selection); } catch (_) {} }
  }

  async function renderCredibility(navigationId) {
    const eventId = state.selectedEvent;
    if (!eventId) { selectionRequired(); return; }
    const [data, event] = await Promise.all([API.getCredibility(eventId), API.getEventDetail(eventId)]);
    if (navigationIsStale('credibility', navigationId)) return;
    const route = routes.credibility;
    const evidence = data.evidence || [];
    const reasons = data.reasons || [];
    const verdict = data.verdict || '待核验';
    $('#pageContainer').innerHTML = `<div class="page-content">
      ${pageHeader(route, '结合公开来源与证据评估可信程度；可信度与事件影响分别展示。', `<button class="primary-button" data-action="generate-report">导出结论</button>`)}
      ${analysisContext(event)}
      <section class="credibility-layout">
        <div>
          <article class="panel score-hero">
            <div class="drawer-event-title" style="margin:0 0 16px;padding:0;border:0">
              <span class="status-pill status-${event.level}">${levelLabel(event.level)}</span>
              <h3>${escapeHTML(event.title)}</h3>
              <p>${escapeHTML(event.company)} · ${escapeHTML(event.code)} · ${escapeHTML(event.time)} · ${escapeHTML(String(event.sourceCount || 0))} 个来源</p>
            </div>
            <span class="eyebrow">可信评估 · ${bandLabel(data.score)}</span>
            <div class="verdict"><i></i>${escapeHTML(verdict)}</div>
            <p class="score-summary">${escapeHTML(data.summary)}</p>
          </article>
          <article class="panel" style="margin-top:14px">${panelHead('判断依据', '公开证据与评估依据')}<div class="factor-list">${reasons.map((item) => `<div class="factor-row"><div class="factor-head"><span>${escapeHTML(item.label)}</span><b>${escapeHTML(item.value)}</b></div><small style="color:var(--muted);font-size:8px">${escapeHTML(item.note || '')}</small></div>`).join('')}</div></article>
        </div>
        <div class="credibility-right"><article class="panel evidence-panel">
          ${panelHead('证据材料', `${evidence.length} 条公开证据`, `<span class="pill" style="--pill-color:#2ee6a6">可直接阅读</span>`)}
          <div class="evidence-grid">${evidence.map((item) => `<article class="evidence-card"><div class="evidence-head"><div class="evidence-source"><i class="source-logo" style="--source-color:${item.color}">${item.logo}</i><span><strong>${escapeHTML(item.source)}</strong><span>${escapeHTML(item.type)}</span></span></div><span class="pill" style="--pill-color:${item.color}">${escapeHTML(item.status)}</span></div><p>${escapeHTML(item.text)}</p><div class="evidence-foot"><span>${escapeHTML(item.time)} 抓取</span><span class="pill" style="--pill-color:${item.color}">已整理</span></div></article>`).join('')}</div>
        </article><section class="reasoning-strip">${reasons.map((item) => `<article class="reason-card"><span>${escapeHTML(item.label)}</span><strong>${escapeHTML(item.value)}</strong><small>${escapeHTML(item.note || '')}</small></article>`).join('')}</section></div>
      </section>
    </div>`;
  }

  async function renderTimeline(navigationId) {
    const eventId = state.selectedEvent;
    if (!eventId) { selectionRequired(); return; }
    const [data, event] = await Promise.all([API.getTimeline(eventId), API.getEventDetail(eventId)]);
    if (navigationIsStale('timeline', navigationId)) return;
    const route = routes.timeline;
    $('#pageContainer').innerHTML = `<div class="page-content">
      ${pageHeader(route, '围绕同一事件持续聚合增量信息，展示公开脉络和影响变化。', `<button class="primary-button" data-action="create-watch">＋ 关注脉络</button>`)}
      ${analysisContext(event)}
      <section class="panel" style="margin-bottom:14px">${panelHead(data.title, `${data.company} · ${data.code} · 当前公开事件样本`, `<span class="pill" style="--pill-color:#2ee6a6">已有结果</span>`)}<div class="timeline-stage">${data.stages.map((stage, index) => `<div class="stage-node ${stage.state}"><span class="stage-dot">${stage.state === 'done' ? '✓' : index + 1}</span><strong>${stage.name}</strong><span>${stage.time}</span></div>`).join('')}</div></section>
      <section class="timeline-layout">
        <article class="panel">${panelHead('事件演化时间线', `${data.items.length} 个关键节点`)}<div class="timeline-feed">${data.items.map((item) => `<div class="timeline-item"><div class="timeline-time"><strong>${item.date}</strong>${item.time}</div><i class="timeline-marker" style="--marker-color:${item.color}"></i><article class="timeline-card"><div class="timeline-card-head"><h4>${item.title}</h4><span class="pill" style="--pill-color:${item.color}">${item.type}</span></div><p>${item.text}</p><div class="timeline-card-foot"><span>▣</span><span>${item.source}</span></div></article></div>`).join('')}</div></article>
        <div>
          <article class="panel impact-card"><span class="eyebrow">影响概览</span><div class="impact-company"><span class="feed-company" style="--logo-color:#2ee6a6">${data.company.slice(0,1)}</span><span><strong>${data.company}</strong><span>${data.code} · ${data.title}</span></span></div><div style="display:flex;align-items:flex-end;justify-content:space-between;margin-top:17px"><span><small style="display:block;color:var(--muted)">影响分（0–100）</small><strong style="font-size:31px">${data.impact}</strong></span><span class="status-pill status-${data.impact >= 90 ? 'critical' : data.impact >= 75 ? 'high' : 'medium'}">${data.impact >= 90 ? '重大' : data.impact >= 75 ? '高影响' : '中影响'}</span></div><div class="impact-meter"><div class="impact-scale"><i class="impact-pin" style="left:${Math.max(0, Math.min(100, Number(data.impact) || 0))}%"></i></div><div class="impact-labels"><span>低</span><span>中</span><span>高</span><span>重大</span></div></div><div class="impact-list">${data.impacts.map((item) => `<div class="impact-row"><span>${item.name}</span><b>${item.value}</b></div>`).join('')}</div></article>
          <article class="panel" style="margin-top:14px;padding:17px"><span class="eyebrow">下一观察点</span><strong style="font-size:11px">后续重点关注</strong><p style="color:var(--muted);font-size:9px;line-height:1.7;margin-bottom:0">关注后续公开披露、业务进展和关联信号的变化。可刷新页面查看当前已载入的数据。</p></article>
        </div>
      </section>
    </div>`;
  }

  async function renderGraph(navigationId) {
    const data = await API.getGraph(state.graphScope);
    if (navigationIsStale('graph', navigationId)) return;
    const route = routes.graph;
    $('#pageContainer').innerHTML = `<div class="page-content">
      ${pageHeader(route, '用关系图发现主体与事件的关联，再在右侧核对节点信息或进入事件分析。', `<select class="select-control graph-scope-select" id="graphScope" aria-label="选择图谱范围"><option value="company" ${state.graphScope === 'company' ? 'selected' : ''}>公司关系</option><option value="event" ${state.graphScope === 'event' ? 'selected' : ''}>事件关系</option><option value="risk" ${state.graphScope === 'risk' ? 'selected' : ''}>风险关系</option></select><button class="ghost-button" data-action="export-graph">⇩ 导出图谱</button><button class="primary-button" data-action="graph-fullscreen">⛶ 全屏探索</button>`)}
      <section class="graph-layout">
        <article class="panel graph-panel">${panelHead('事件关系图谱', '点击节点查看详情；拖拽节点调整位置，拖拽空白区域平移', `<span class="mini-tag">${state.graphScope === 'risk' ? '风险关系' : state.graphScope === 'event' ? '事件关系' : '公司关系'}</span>`)}<div class="graph-canvas-wrap graph-force-shell"><div class="graph-toolbar"><div class="graph-legend" aria-label="图谱图例"><span><i style="--c:#2ee6a6"></i>公司</span><span><i style="--c:#4aa8ff"></i>事件</span><span><i style="--c:#9677ff"></i>类型</span><span><i style="--c:#ff5d73"></i>风险</span></div><div class="graph-zoom"><button aria-label="放大图谱" data-graph-action="in">＋</button><button aria-label="缩小图谱" data-graph-action="out">−</button><button aria-label="重置图谱" data-graph-action="reset">⌂</button></div></div><div class="graph-interaction-hint">节点可拖动 · 点击节点显示关联关系 · 滚轮缩放</div><div class="graph-edge-count">${data.nodes.length} 个节点 · ${data.links.length} 条关系</div><canvas id="knowledgeGraph" data-graph-interactive="true" aria-label="事件关系图谱"></canvas></div></article>
        <aside class="graph-sidebar"><article class="panel node-info" id="nodeInfo">${nodeInfoHTML(data.selected)}</article><article class="panel graph-relations-panel">${panelHead('主要关联', '当前节点的一跳关系')}<div class="relation-list">${data.relations.length ? data.relations.map((item) => `<div class="relation-row"><b>${escapeHTML(item.name)}</b><span>${escapeHTML(item.relation)}</span></div>`).join('') : '<div class="graph-empty-note">暂无可展示的关联节点</div>'}</div></article></aside>
      </section>
    </div>`;
    state.graphController = Charts.graph($('#knowledgeGraph'), data, (node) => {
      $('#nodeInfo').innerHTML = nodeInfoHTML(node, node.color);
    });
  }

  function nodeInfoHTML(node = {}, color = '#2f75f6') {
    const name = node.name || node.label || '未选择节点';
    const type = ({company:'上市公司', event:'事件', industry:'事件类型', technology:'处理环节', risk:'风险', policy:'政策'})[node.type] || node.typeLabel || node.type || '';
    return `<span class="eyebrow">当前图谱节点</span><div class="node-icon-large" style="color:${color}">${escapeHTML(name.slice(0,1))}</div><h3>${escapeHTML(name)}</h3><span>${escapeHTML(node.code || '')} · ${escapeHTML(type)}</span><div class="node-stats"><div><span>直接关联事件</span><b>${escapeHTML(node.events ?? '未提供')}</b></div><div><span>直接关联节点</span><b>${escapeHTML(node.relations ?? '未提供')}</b></div><div><span>事件可信度</span><b>${bandLabel(node.credibility)}</b></div><div><span>事件影响级别</span><b>${escapeHTML(node.risk ?? '未提供')}</b></div></div><p class="node-note">${escapeHTML(node.summary || '统计范围为当前展示的图谱。')}</p>${node.eventId ? `<button class="primary-button" data-nav="credibility" data-analyze-id="${escapeHTML(node.eventId)}">分析此事件 →</button>` : ''}`;
  }

  async function renderAlerts(navigationId = state.navigationId) {
    if (state.route !== 'alerts') return;
    const requestId = ++state.alertRequest;
    const filters = { ...state.alertFilters };
    const [allAlerts, rules, selected] = await Promise.all([API.getAlerts(), API.getAlertRules(), filters.eventId ? API.getEventDetail(filters.eventId) : null]);
    if (navigationIsStale('alerts', navigationId) || requestId !== state.alertRequest) return;
    state.alertRule = rules;
    syncAlertBadges(allAlerts);
    const alerts = allAlerts.filter(item => (filters.level === 'all' || item.level === filters.level) && (filters.status === 'all' || item.status === filters.status) && (!filters.eventId || item.eventId === filters.eventId));
    $('#pageContainer').innerHTML = `<div class="page-content">
      ${pageHeader(routes.alerts, '依据当前规则筛选已有事件，确认后进入跟踪；确认状态会保留。', '<button class="ghost-button" data-action="view-watches">我的关注</button><button class="primary-button" data-action="alert-settings">设置预警规则</button>')}
      ${selected ? analysisContext(selected) : ''}
      <section class="alert-summary">${[[allAlerts.filter(i=>i.level==='critical').length,'重大预警','#ff5d73','!'],[allAlerts.filter(i=>i.level==='high').length,'高影响预警','#ffb648','↗'],[allAlerts.filter(i=>i.status==='待处置').length,'待处置','#4aa8ff','◎'],[allAlerts.filter(i=>i.status==='跟踪中').length,'已确认 · 跟踪中','#2ee6a6','✓']].map(([value,label,color,icon])=>`<article class="panel alert-stat"><i class="alert-stat-icon" style="--stat-color:${color}">${icon}</i><span><b>${value}</b><span>${label}</span></span></article>`).join('')}</section>
      <p class="rule-summary">当前规则：影响分 ≥ ${rules.high} 进入预警，≥ ${rules.critical} 为重大；最低可信度 ${rules.credibility} / 100。平台内展示。</p>
      <section class="panel toolbar"><div class="filter-group">
        <select class="select-control" id="alertLevelFilter" aria-label="预警级别">${[['all','全部预警级别'],['critical','重大预警'],['high','高影响预警']].map(([value,label])=>`<option value="${value}" ${filters.level===value?'selected':''}>${label}</option>`).join('')}</select>
        <select class="select-control" id="alertStatusFilter" aria-label="处置状态">${[['all','全部处置状态'],['待处置','待处置'],['跟踪中','跟踪中']].map(([value,label])=>`<option value="${value}" ${filters.status===value?'selected':''}>${label}</option>`).join('')}</select>
        ${filters.eventId ? '<span class="mini-tag">仅当前事件</span>' : ''}</div><div class="filter-group"><span class="result-count">共 ${alerts.length} 条</span><button class="link-button" data-action="reset-alert-filters">查看全部 / 重置</button></div></section>
      <section>${alerts.length ? alerts.map(alert=>`<article class="panel alert-card" data-alert-id="${escapeHTML(alert.id)}" style="--alert-color:${alert.color}"><i class="alert-accent"></i><div class="alert-main"><i class="alert-level-icon">${alert.icon}</i><div class="alert-copy"><h3>${escapeHTML(alert.title)}</h3><p>${escapeHTML(alert.company)} · ${escapeHTML(alert.reason)}</p></div></div><div class="alert-meta"><div><span>预警级别</span><b>${levelLabel(alert.level)}</b></div><div><span>事件时间</span><b>${escapeHTML(alert.time)}</b></div><span class="status-pill status-${alert.level}">${escapeHTML(alert.status)}</span><button class="ghost-button" data-nav="credibility" data-analyze-id="${escapeHTML(alert.eventId)}">查看证据</button>${alert.status==='待处置' ? `<button class="secondary-button" data-action="ack-alert" data-alert-id="${escapeHTML(alert.id)}">确认处置</button>` : ''}</div></article>`).join('') : '<div class="panel empty-state"><strong>暂无符合条件的预警</strong><span>当前事件可能未达到规则阈值，也可重置筛选查看其他预警。</span><button class="secondary-button" data-action="reset-alert-filters">重置筛选</button></div>'}</section>
    </div>`;
  }

  function syncAlertBadges(alerts) {
    const pending = alerts.filter(item => item.status === '待处置').length;
    routes.alerts.badge = pending;
    renderNavigation();
    const badge = $('#notificationButton i');
    badge.textContent = pending > 99 ? '99+' : String(pending);
    badge.hidden = pending === 0;
  }

  function renderIntegrationAlertCard(card) {
    const severity = card.severity || 'S0';
    const level = card.level || ({ S4: 'critical', S3: 'high', S2: 'medium', S1: 'low', S0: 'low' })[severity] || 'medium';
    const color = card.color || ({ S4: '#ff5d73', S3: '#ffb648', S2: '#4aa8ff', S1: '#9677ff', S0: '#4ce3ef' })[severity] || '#ffb648';
    const icon = card.icon || ({ S4: '!', S3: '↗', S2: '◇', S1: '•', S0: '·' })[severity] || '◇';
    const status = card.status || '已完成';
    const title = card.title || '';
    const eventName = card.eventName || '';
    const scope = card.scope || '';
    const reason = card.reason || '';
    const safeReason = /QWK|F1|Acc|Macro|P95|Bootstrap|评分|可信|official|解析率|原型|任务|模型/i.test(reason) ? '' : reason;
    return `<article class="panel process-alert-card" style="--alert-color:${color}">
      <div style="height:5px;background:${color}"></div>
      <div style="padding:16px 18px">
        <div class="process-alert-top">
          <div class="process-alert-main">
            <i class="alert-level-icon" style="--alert-color:${color};width:42px;height:42px">${icon}</i>
            <div class="process-alert-copy">
              <span class="eyebrow">重点提醒</span>
              <h3>${escapeHTML(title)}</h3>
              <p>${escapeHTML(eventName)} · ${escapeHTML(scope)}</p>
            </div>
          </div>
          <span class="status-pill status-${level}" style="flex:0 0 auto">${escapeHTML(status)}</span>
        </div>
        <div class="process-alert-tags">
          <span class="mini-tag">${escapeHTML(levelLabel(level))}</span>
          <span class="mini-tag">${escapeHTML(scope || '公开范围')}</span>
          <span class="mini-tag">${escapeHTML(card.urgency || '持续关注')}</span>
        </div>
        <div class="key-value-grid process-alert-grid">
          <div class="key-value"><span>公开范围</span><b>${escapeHTML(scope || '—')}</b></div>
          <div class="key-value"><span>提醒等级</span><b>${escapeHTML(levelLabel(level))}</b></div>
          <div class="key-value"><span>关联事件</span><b>${escapeHTML(eventName || '—')}</b></div>
          <div class="key-value"><span>关注点</span><b>${escapeHTML(card.urgency || '—')}</b></div>
        </div>
        <p class="process-stage-note" style="margin-top:14px">${escapeHTML(safeReason || '系统已完成公开信号汇总。')}</p>
      </div>
    </article>`;
  }

  function renderIntegrationCaseCard(item) {
    const task3 = item.task3_event_interface || {};
    const task4 = item.task4_formal_chain || {};
    return `<article class="reason-card">
      <span>${escapeHTML(item.theme || '案例')}</span>
      <strong>${escapeHTML(task3StateLabelLocal(task3.current_state_type || 'unknown'))}</strong>
      <small>${escapeHTML(formatPercent(task3.trust_score || 0))}</small>
      <small>${escapeHTML(task4.validity_model || '—')} / ${escapeHTML(task4.direction_model || '—')}</small>
      <small>${escapeHTML(item.note || '')}</small>
    </article>`;
  }

  function renderIntegrationView() {
    const data = activeProcessData();
    if (!data?.entries?.length) return;
    const queueItems = data.queue || [];
    const stageCards = (data.stageOrder || []).filter((item) => item.key !== 'ingest');
    const selectedQueueItem = state.processDemo.activeIndex >= 0 ? queueItems[state.processDemo.activeIndex] || null : null;
    const selectedEntry = selectedQueueItem ? (data.entries || []).find((item) => item.id === selectedQueueItem.id) || null : null;
    const selectedStage = currentProcessStage();
    const selectedStageIndex = selectedStage ? state.processDemo.stageIndex : -1;
    const selectedStatus = selectedQueueItem && selectedStage ? processStageState(selectedStageIndex) : 'idle';
    const contextTitle = selectedQueueItem?.title || '请选择一个待处理事件';
    const contextMeta = selectedQueueItem ? `${selectedQueueItem.company || '未知主体'} · ${selectedQueueItem.code || '—'} · ${selectedQueueItem.sourceCount || 0} 个来源` : '选择事件后从接入阶段开始处理';
    let resultMarkup = '<div class="empty-state process-stage-empty"><strong>请选择一个事件开始处理</strong><span>选择事件后，系统会从“进入队列”开始，按顺序解锁后续任务。</span></div>';
    if (selectedQueueItem && selectedEntry && selectedStage) {
      if (selectedStatus === 'processing') resultMarkup = processStageProcessingMarkup(selectedStage);
      else if (selectedStatus === 'completed') resultMarkup = processStageMarkup(selectedEntry, selectedStage);
      else if (selectedStatus === 'ready') resultMarkup = processStageReadyMarkup(selectedEntry, selectedStage, selectedStageIndex);
      else resultMarkup = `<div class="process-simulation-locked"><span class="eyebrow">尚未解锁</span><h3>${escapeHTML(selectedStage.label)}</h3><p>请先完成前置阶段，系统会自动开放当前任务。</p></div>`;
    }

    $('#pageContainer').innerHTML = `<div class="page-content process-page">
      ${pageHeader(routes.integration, '选择一个事件，按阶段查看处理结果。', '<button class="ghost-button" data-action="process-reset">↺ 重新开始</button><button class="ghost-button" data-nav="dashboard">⌂ 返回概览</button>')}
      <section class="panel process-context-bar">
        <div class="process-context-copy"><span class="eyebrow">任务链路仿真处理</span><h2>${escapeHTML(contextTitle)}</h2><p>${escapeHTML(contextMeta)}</p></div>
        <div class="process-context-controls"><label class="process-control-label" for="processEventSelect">待处理事件</label><select class="select-control process-select" id="processEventSelect" aria-label="选择待处理事件"><option value="">请选择事件</option>${queueItems.map((item, index) => `<option value="${index}" ${index === state.processDemo.activeIndex ? 'selected' : ''}>${escapeHTML(item.title)} · ${escapeHTML(item.company)} · ${escapeHTML(String(item.sourceCount || 0))} 源</option>`).join('')}</select></div>
      </section>
      <section class="process-workspace">
        <aside class="panel process-stage-rail" aria-label="任务阶段导航">
          <div class="process-rail-head"><span class="eyebrow">处理阶段</span><strong>${selectedQueueItem ? `${Math.max(0, state.processDemo.completedStage + 1)} / ${data.stageOrder.length} 阶段完成` : '等待选择事件'}</strong></div>
          <div class="process-rail-list">${stageCards.map((stage, index) => {
            const stageIndex = (data.stageOrder || []).findIndex((item) => item.key === stage.key);
            const active = stageIndex === selectedStageIndex;
            const status = selectedQueueItem ? processStageState(stageIndex) : 'locked';
            const disabled = !selectedQueueItem || status === 'locked' || status === 'processing';
            return `<button class="process-rail-item ${active ? 'active' : ''} process-rail-${status} ${disabled ? 'disabled' : ''}" data-process-stage="${stageIndex}" ${disabled ? 'disabled' : ''} style="--stage-color:${stage.color || '#4aa8ff'}"><span class="process-rail-index">${status === 'completed' ? '✓' : index + 1}</span><span class="process-rail-copy"><b>${escapeHTML(stage.label)}</b><small>${escapeHTML(processStageBrief(stage.key))}</small></span><span class="process-rail-state">${escapeHTML(selectedQueueItem && status === 'completed' ? processStageResult(stage.key, selectedEntry) : selectedQueueItem ? processStageStateLabel(status) : '待选')}</span></button>`;
          }).join('')}</div>
          <div class="process-rail-note">每完成一个阶段才会解锁下一阶段；已完成阶段可以回看，未解锁阶段不会提前展示结果。</div>
        </aside>
        <main class="panel process-current-result" aria-live="polite">
          <div class="process-result-head"><div><span class="eyebrow">当前处理阶段</span><h2>${escapeHTML(selectedStage?.label || '尚未选择事件')}</h2><p>${escapeHTML(selectedEntry ? `${selectedEntry.title} · ${selectedEntry.company}` : '选择事件后，先从进入队列开始。')}</p></div><span class="process-result-status process-status-${selectedStatus}">${escapeHTML(selectedQueueItem ? (selectedStatus === 'idle' ? '待选择' : processStageStateLabel(selectedStatus)) : '待选择')}</span></div>
          <div class="process-stage-main">${resultMarkup}</div>
        </main>
      </section>
    </div>`;
  }

  async function renderIntegration(navigationId) {
    const data = await API.getProcessDemo(5);
    if (navigationIsStale('integration', navigationId)) return;
    state.processDemo.data = data;
    if (!data?.entries?.length) {
      $('#pageContainer').innerHTML = '<div class="empty-state"><strong>暂无演示样本</strong><span>请确认后端已启动并能读取现有样本</span></div>';
      return;
    }
    state.processDemo.activeIndex = Math.min(state.processDemo.activeIndex, data.entries.length - 1);
    state.processDemo.stageIndex = Math.min(state.processDemo.stageIndex, data.stageOrder.length - 1);
    renderIntegrationView();
  }

  async function openEventDrawer(id) {
    const requestId = ++state.drawerRequest;
    const navigationId = state.navigationId;
    try {
    const event = await API.getEventDetail(id);
    if (requestId !== state.drawerRequest || navigationId !== state.navigationId) return;
    $('#drawerContent').innerHTML = `<div class="drawer-event-title"><span class="status-pill status-${event.level}">${levelLabel(event.level)}</span><h3>${escapeHTML(event.title)}</h3><p>${escapeHTML(event.company)} · ${escapeHTML(event.code)} · ${escapeHTML(event.time)} · ${escapeHTML(String(event.sourceCount || 0))} 个来源</p></div><div class="drawer-score-grid"><div class="drawer-score"><span>可信度（0–100）</span><b style="--score-color:var(--primary)">${escapeHTML(bandLabel(event.credibility))}</b></div><div class="drawer-score"><span>情绪方向</span><b style="--score-color:var(--warning)">${escapeHTML(sentimentLabel(event.sentiment))}</b></div><div class="drawer-score"><span>影响级别</span><b style="--score-color:var(--blue)">${escapeHTML(levelLabel(event.level))}</b></div></div><section class="drawer-section"><h4>事件摘要</h4><p style="color:var(--muted);font-size:10px;line-height:1.8">${escapeHTML(event.summary)}</p></section><section class="drawer-section"><h4>公开要素</h4><div class="key-value-grid"><div class="key-value"><span>事件主体</span><b>${escapeHTML(event.entity)}</b></div><div class="key-value"><span>影响对象</span><b>${escapeHTML(event.object)}</b></div><div class="key-value"><span>事件类型</span><b>${escapeHTML(event.type)}</b></div><div class="key-value"><span>影响方向</span><b>${escapeHTML(event.direction)}</b></div><div class="key-value"><span>当前状态</span><b>${escapeHTML(event.status)}</b></div><div class="key-value"><span>来源数量</span><b>${escapeHTML(String(event.sourceCount || 0))} 个</b></div></div></section><section class="drawer-section"><h4>关键词</h4><p style="color:var(--muted);font-size:10px;line-height:1.75">${escapeHTML(event.sourceText)}</p><div style="display:flex;gap:6px;flex-wrap:wrap">${event.keywords.map(keyword=>`<span class="mini-tag">${escapeHTML(keyword)}</span>`).join('')}</div></section><div class="drawer-actions"><button class="secondary-button" data-nav="timeline" data-analyze-id="${escapeHTML(event.id)}">查看事件脉络</button><button class="primary-button" data-nav="credibility" data-analyze-id="${escapeHTML(event.id)}">查看可信评估</button></div>`;
    $('#detailDrawer').classList.add('open'); $('#drawerBackdrop').classList.add('open'); $('#detailDrawer').setAttribute('aria-hidden','false');
    } catch (error) { if (requestId === state.drawerRequest) showToast(error.message, true); }
  }

  function closeDrawer() {
    state.drawerRequest += 1;
    $('#detailDrawer').classList.remove('open'); $('#drawerBackdrop').classList.remove('open'); $('#detailDrawer').setAttribute('aria-hidden','true');
  }

  function openModal(content) {
    $('#modalContent').innerHTML = content; $('#modalBackdrop').classList.add('open');
  }

  function closeModal() { $('#modalBackdrop').classList.remove('open'); }

  function openWatchModal() {
    openModal(`<h2 id="modalTitle">保存事件关注</h2><p>订阅公司、事件类型或自定义事件，系统将在满足阈值时自动推送预警。</p><form id="watchForm"><div class="form-error" role="alert"></div><div class="form-field"><label>监控对象</label><input name="target" required placeholder="如：上市公司名称 / 证券代码 / 事件类型"></div><div class="form-grid"><div class="form-field"><label>监控范围</label><select name="scope"><option>上市公司事件</option><option>行业事件</option><option>产业链事件</option></select></div><div class="form-field"><label>最低影响级别</label><select name="level"><option>中影响及以上</option><option>高影响及以上</option><option>仅重大事件</option></select></div></div><div class="form-field"><label>关键词（可选）</label><input name="keywords" placeholder="用逗号分隔，如：增持, 处罚, 业务拓展"></div><div class="form-field"><label>关注说明</label><textarea name="note" placeholder="记录监控目的或需要特别关注的信息"></textarea></div><div class="modal-actions"><button type="button" class="ghost-button" data-action="close-modal">取消</button><button type="submit" class="primary-button">保存关注</button></div></form>`);
  }

  async function openAlertSettingsModal() {
    const rules = await API.getAlertRules();
    state.alertRule = rules;
    openModal(`<h2 id="modalTitle">预警规则设置</h2><p>规则保存后立即应用于预警列表。仅在平台内展示，不发送邮件。</p><form id="alertSettingsForm"><div class="form-grid"><div class="form-field"><label>高影响预警阈值（0–100）</label><input name="high" type="number" min="0" max="99" step="any" required value="${rules.high}"></div><div class="form-field"><label>重大预警阈值（0–100）</label><input name="critical" type="number" min="1" max="100" step="any" required value="${rules.critical}"></div></div><div class="form-field"><label>最低可信度（0 表示不限）</label><input name="credibility" type="number" min="0" max="100" step="any" required value="${rules.credibility}"></div><p>提高最低可信度会排除部分低可信事件，请结合风险关注需求设置。</p><div class="form-error" role="alert"></div><div class="modal-actions"><button type="button" class="ghost-button" data-action="close-modal">取消</button><button type="submit" class="primary-button">保存并应用</button></div></form>`);
  }

  async function openWatches() {
    const watches = await API.getWatches();
    openModal(`<h2 id="modalTitle">我的关注</h2><p>已保存的关注条件，可用于快速检索。当前不自动发送通知。</p><div class="watch-list">${watches.length ? watches.map(item=>`<article class="watch-item"><strong>${escapeHTML(item.target)}</strong><p>${escapeHTML(item.scope)} · ${escapeHTML(item.level)}</p><small>${escapeHTML(item.keywords || '未设置关键词')}</small><p>${escapeHTML(item.note || '')}</p><button class="link-button" data-action="search-watch" data-query="${escapeHTML(item.target)}">查找相关事件 →</button></article>`).join('') : '<div class="empty-state"><strong>暂无关注条件</strong><span>可以从事件列表保存关注对象。</span></div>'}</div><div class="modal-actions"><button class="primary-button" data-action="create-watch">新增关注</button></div>`);
  }

  async function openAlertDetail(id) {
    const alerts = await API.getAlerts();
    const alert = alerts.find((item) => item.id === id);
    if (!alert) return;
    openModal(`<h2 id="modalTitle">预警详情</h2><p>${escapeHTML(alert.company)}</p><div style="display:flex;align-items:center;gap:12px;margin:18px 0"><i class="alert-level-icon" style="--alert-color:${alert.color}">${alert.icon}</i><div><strong style="font-size:14px">${escapeHTML(alert.title)}</strong><div style="margin-top:5px"><span class="status-pill status-${alert.level}">${levelLabel(alert.level)}</span></div></div></div><div class="key-value-grid"><div class="key-value"><span>影响级别</span><b>${levelLabel(alert.level)}</b></div><div class="key-value"><span>当前状态</span><b>${alert.status}</b></div><div class="key-value"><span>触发时间</span><b>${alert.time}</b></div><div class="key-value"><span>提醒原因</span><b>${escapeHTML(alert.reason)}</b></div></div><div class="modal-actions"><button class="ghost-button" data-nav="credibility" data-analyze-id="${escapeHTML(alert.eventId)}">查看事件证据</button>${alert.status === '待处置' ? `<button class="primary-button" data-action="ack-alert" data-alert-id="${alert.id}">确认处置</button>` : ''}</div>`);
  }

  async function openSearch(mode = 'preview') {
    state.searchMode = typeof mode === 'string' ? mode : 'preview';
    $('#commandBackdrop').classList.add('open'); $('#commandInput').value = ''; $('#commandInput').focus();
    await updateSearchResults('');
  }

  function closeSearch() { state.searchRequest += 1; $('#commandBackdrop').classList.remove('open'); }

  async function chooseSearchResult(id) {
    const mode = state.searchMode;
    closeSearch();
    if (mode === 'select') {
      state.selectedEvent = id;
      if (state.route === 'alerts') state.alertFilters.eventId = id;
      await navigate(state.route);
    } else await openEventDrawer(id);
  }

  async function updateSearchResults(query) {
    const requestId = ++state.searchRequest;
    $('#commandResults').innerHTML = '<div class="empty-state"><span>正在搜索…</span></div>';
    try {
      const results = await API.search(query);
      if (requestId !== state.searchRequest || !$('#commandBackdrop').classList.contains('open')) return;
      $('#commandResults').innerHTML = `<div class="command-label">${state.searchMode === 'select' ? '选择当前分析事件' : query ? '搜索结果' : '近期重点事件'}</div>${results.length ? results.map((item,index)=>`<button class="command-item ${index===0?'active':''}" data-event-id="${escapeHTML(item.id)}"><i class="command-item-icon">◫</i><span class="command-item-copy"><strong>${escapeHTML(item.title)}</strong><span>${escapeHTML(item.meta)}</span></span><span class="status-pill status-${item.level}">${levelLabel(item.level)}</span></button>`).join('') : '<div class="empty-state"><strong>未找到相关结果</strong><span>尝试输入公司简称、证券代码或事件关键词</span></div>'}`;
    } catch (error) {
      if (requestId !== state.searchRequest) return;
      $('#commandResults').innerHTML = `<div class="empty-state"><strong>搜索失败</strong><span>${escapeHTML(error.message)}</span><button class="secondary-button" data-action="retry-search">重试</button></div>`;
    }
  }

  async function renderNotifications() {
    try {
      const alerts = await API.getAlerts();
      syncAlertBadges(alerts);
      const pending = alerts.filter(item => item.status === '待处置');
      $('#notificationList').innerHTML = pending.length ? pending.slice(0, 3).map(item=>`<button class="notification-item" data-alert-id="${escapeHTML(item.id)}" style="--notice-color:${item.color}"><i></i><div><strong>${levelLabel(item.level)}预警</strong><p>${escapeHTML(item.title)}</p></div><time>${escapeHTML(item.time)}</time></button>`).join('') : '<div class="empty-state"><strong>暂无待处置预警</strong></div>';
    } catch (_) {
      $('#notificationList').innerHTML = '<div class="notification-item"><div><strong>通知加载失败</strong><p>请确认后端服务已启动</p></div></div>';
    }
  }

  function showToast(message, failed = false) {
    const toast = document.createElement('div'); toast.className=`toast${failed ? ' toast-error' : ''}`; toast.innerHTML=`<i>${failed ? '!' : '✓'}</i><span>${escapeHTML(message)}</span>`;
    $('#toastRegion').appendChild(toast); setTimeout(()=>toast.remove(),3200);
  }

  async function exportEventsCSV() {
    const events = await API.getEvents(state.eventFilters);
    const rows = [['事件ID','事件标题','关联公司','证券代码','事件类型','可信度（0–100）','影响级别','状态'], ...events.map((item) => [item.id,item.title,item.company,item.code,item.type,bandLabel(item.credibility),levelLabel(item.level),item.status])];
    const csv = `\uFEFF${rows.map((row) => row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(',')).join('\r\n')}`;
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a'); link.href = url; link.download = `金融事件清单-${new Date().toISOString().slice(0,10)}.csv`; link.click();
    URL.revokeObjectURL(url); showToast('事件清单已导出');
  }

  function exportGraphImage() {
    const canvas = $('#knowledgeGraph');
    if (!canvas) return;
    const link = document.createElement('a'); link.href = canvas.toDataURL('image/png'); link.download = '事件知识图谱.png'; link.click();
    showToast('知识图谱图片已导出');
  }

  async function enterGraphFullscreen() {
    const panel = $('.graph-panel');
    if (panel?.requestFullscreen) { await panel.requestFullscreen(); }
    else showToast('当前浏览器不支持全屏模式');
  }

  async function updateDataStatus() {
    const indicator = $('.stream-indicator span');
    try {
      const status = await API.getDataStatus();
      indicator.textContent = API.config.USE_MOCK ? '演示样本' : '已载入项目数据';
      indicator.parentElement.title = `数据版本：${status.dataVersionAt || '未提供'}；载入时间：${status.loadedAt || '未提供'}`;
    } catch (_) { indicator.textContent = '数据连接待检查'; }
  }

  function closePanels() {
    $('#notificationPanel').classList.remove('open'); $('#sidebar').classList.remove('open');
  }

  document.addEventListener('click', async (event) => {
    const nav = event.target.closest('[data-nav]');
    if (nav) {
      event.preventDefault();
      if (nav.dataset.analyzeId) state.selectedEvent = nav.dataset.analyzeId;
      if (nav.dataset.nav === 'alerts') state.alertFilters.eventId = nav.dataset.eventFilter || '';
      closeModal(); await navigate(nav.dataset.nav); return;
    }
    const eventTarget = event.target.closest('[data-event-id]');
    if (eventTarget && !eventTarget.closest('#commandResults')) { openEventDrawer(eventTarget.dataset.eventId); return; }
    const actionTarget = event.target.closest('[data-action]');
    const alertTarget = event.target.closest('[data-alert-id]');
    if (alertTarget && !actionTarget) { closePanels(); await perform(null, () => openAlertDetail(alertTarget.dataset.alertId)); return; }
    if (!actionTarget) return;
    const action = actionTarget.dataset.action;
    if (action === 'close-drawer') closeDrawer();
    if (action === 'close-modal') closeModal();
    if (action === 'create-watch') openWatchModal();
    if (action === 'refresh') { if (await navigate(state.route, false)) { await renderNotifications(); await updateDataStatus(); showToast('已重新载入当前数据'); } }
    if (action === 'export') await perform(actionTarget, exportEventsCSV);
    if (action === 'export-graph') exportGraphImage();
    if (action === 'generate-report') { showToast('已打开浏览器打印，可保存为 PDF 报告'); window.print(); }
    if (action === 'graph-fullscreen') await perform(actionTarget, enterGraphFullscreen);
    if (action === 'select-event') openSearch('select');
    if (action === 'retry-search') updateSearchResults($('#commandInput').value);
    if (action === 'view-watches') await perform(actionTarget, openWatches);
    if (action === 'search-watch') { state.eventPage = 1; state.eventFilters = { query: actionTarget.dataset.query, level: 'all', sentiment: 'all' }; closeModal(); await navigate('events'); }
    if (action === 'event-prev' || action === 'event-next') { state.eventPage += action === 'event-next' ? 1 : -1; await perform(actionTarget, () => renderEvents()); }
    if (action === 'reset-filters') { state.eventPage = 1; state.eventFilters = { query: '', level: 'all', sentiment: 'all' }; await perform(null, () => renderEvents()); }
    if (action === 'reset-alert-filters') { state.alertFilters = { level: 'all', status: 'all', eventId: '' }; await navigate('alerts'); }
    if (action === 'process-start') {
      const stageIndex = Number(actionTarget.dataset.stageIndex);
      if (Number.isFinite(stageIndex)) startProcessStage(stageIndex);
    }
    if (action === 'process-reset') {
      stopProcessPlayback();
      state.processDemo.activeIndex = -1;
      state.processDemo.stageIndex = -1;
      if (state.route === 'integration') renderIntegrationView();
    }
    if (action === 'process-toggle') {
      if (state.processDemo.timer) {
        stopProcessPlayback();
      } else {
        startProcessPlayback();
      }
      if (state.route === 'integration') renderIntegrationView();
    }
    if (action === 'process-prev') {
      moveProcessPlayback(-1);
    }
    if (action === 'process-next') {
      moveProcessPlayback(1);
    }
    if (action === 'ack-alert') await perform(actionTarget, async () => {
      await API.acknowledgeAlert(actionTarget.dataset.alertId);
      closeModal(); showToast('预警已确认，状态已保存');
      await renderNotifications();
      if (state.route === 'alerts') await renderAlerts();
    });
    if (action === 'alert-settings') await perform(actionTarget, openAlertSettingsModal);
  });

  document.addEventListener('click', (event) => {
    const processItem = event.target.closest('[data-process-index]');
    if (processItem) {
      const index = Number(processItem.dataset.processIndex);
      if (Number.isFinite(index) && state.processDemo.data?.entries?.length) {
        setProcessSelection(index, -1);
      }
      return;
    }
    const processStage = event.target.closest('[data-process-stage]');
    if (processStage) {
      if (state.processDemo.activeIndex < 0) return;
      const index = Number(processStage.dataset.processStage);
      if (Number.isFinite(index) && state.processDemo.data?.stageOrder?.length) {
        setProcessSelection(state.processDemo.activeIndex, index);
      }
    }
  });

  document.addEventListener('change', (event) => {
    if (event.target.id === 'levelFilter') { state.eventPage = 1; state.eventFilters.level=event.target.value; perform(null, () => renderEvents()); }
    if (event.target.id === 'sentimentFilter') { state.eventPage = 1; state.eventFilters.sentiment=event.target.value; perform(null, () => renderEvents()); }
    if (event.target.id === 'alertLevelFilter') { state.alertFilters.level=event.target.value; perform(null, () => renderAlerts()); }
    if (event.target.id === 'alertStatusFilter') { state.alertFilters.status=event.target.value; perform(null, () => renderAlerts()); }
    if (event.target.id === 'graphScope') { state.graphScope = event.target.value || 'company'; perform(null, () => navigate('graph', false)); }
    if (event.target.id === 'processEventSelect') {
      const index = event.target.value === '' ? -1 : Number(event.target.value);
      if (Number.isFinite(index) && state.processDemo.data?.entries?.length) {
        setProcessSelection(index, -1);
      } else {
        setProcessSelection(-1, -1);
      }
    }
    if (event.target.id === 'processStageSelect') {
      if (state.processDemo.activeIndex < 0) return;
      const index = event.target.value === '' ? -1 : Number(event.target.value);
      if (Number.isFinite(index) && state.processDemo.data?.stageOrder?.length) {
        setProcessSelection(state.processDemo.activeIndex, index);
      } else {
        setProcessSelection(state.processDemo.activeIndex, -1);
      }
    }
  });

  document.addEventListener('input', (event) => {
    if (event.target.id === 'eventSearch') {
      state.eventPage = 1; state.eventFilters.query = event.target.value;
      clearTimeout(event.target._timer); event.target._timer=setTimeout(()=>perform(null, () => renderEvents()),260);
    }
    if (event.target.id === 'commandInput') {
      state.searchRequest += 1;
      clearTimeout(event.target._timer); event.target._timer=setTimeout(()=>{ if ($('#commandBackdrop').classList.contains('open')) updateSearchResults(event.target.value); },120);
    }
  });

  document.addEventListener('submit', async (event) => {
    const form = event.target;
    if (!['watchForm', 'alertSettingsForm'].includes(form.id)) return;
    event.preventDefault();
    const button = $('button[type="submit"]', form);
    const errorBox = $('.form-error', form);
    errorBox.textContent = '';
    await perform(button, async () => {
      const payload = Object.fromEntries(new FormData(form));
      try {
        if (form.id === 'watchForm') {
          if (!payload.target.trim()) throw new Error('请输入关注对象');
          await API.createWatch(payload);
          closeModal(); showToast('关注条件已保存，可在“我的关注”中查看');
        } else {
          if (Number(payload.high) >= Number(payload.critical)) throw new Error('重大预警阈值必须高于高影响预警阈值');
          state.alertRule = await API.saveAlertRules(payload);
          closeModal(); showToast('规则已保存并应用');
          await renderNotifications();
          if (state.route === 'alerts') await renderAlerts();
        }
      } catch (error) { errorBox.textContent = error.message; throw error; }
    });
  });

  document.addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); openSearch(); }
    if ($('#commandBackdrop').classList.contains('open') && ['ArrowDown','ArrowUp','Enter'].includes(event.key)) {
      const items = $$('.command-item');
      if (items.length) {
        event.preventDefault();
        const current = Math.max(0, items.findIndex(item => item.classList.contains('active')));
        if (event.key === 'Enter') chooseSearchResult(items[current].dataset.eventId);
        else {
          const next = (current + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
          items.forEach((item, index) => item.classList.toggle('active', index === next));
          items[next].scrollIntoView({ block: 'nearest' });
        }
      }
      return;
    }
    if (event.key === 'Escape') { closeSearch(); closeModal(); closeDrawer(); }
    if (event.key === 'Enter' && event.target.matches('[data-event-id][role="button"]')) openEventDrawer(event.target.dataset.eventId);
  });

  $('#globalSearch').addEventListener('click', openSearch);
  $('#commandBackdrop').addEventListener('click', (event) => { if (event.target === $('#commandBackdrop')) closeSearch(); });
  $('#modalBackdrop').addEventListener('click', (event) => { if (event.target === $('#modalBackdrop')) closeModal(); });
  $('#drawerBackdrop').addEventListener('click', closeDrawer);
  $('#commandResults').addEventListener('click', (event) => { const item=event.target.closest('[data-event-id]'); if(item){ chooseSearchResult(item.dataset.eventId); } });
  $('#menuButton').addEventListener('click', () => $('#sidebar').classList.toggle('open'));
  $('#notificationButton').addEventListener('click', (event) => { event.stopPropagation(); $('#notificationPanel').classList.toggle('open'); });
  
  document.body.addEventListener('click', (event)=>{ if (!event.target.closest('#notificationPanel')) $('#notificationPanel').classList.remove('open'); });
  $('#themeToggle').addEventListener('click', () => {
    state.theme = document.body.classList.toggle('light-theme') ? 'light' : 'dark'; preferences.set('finsight-theme',state.theme); showToast(state.theme==='light'?'已切换浅色模式':'已切换深色模式'); setTimeout(()=>navigate(state.route,false),50);
  });
  document.addEventListener('click', (event)=>{
    const control=event.target.closest('[data-graph-action]'); if(!control||!state.graphController)return;
    if(control.dataset.graphAction==='in')state.graphController.zoom(.12); if(control.dataset.graphAction==='out')state.graphController.zoom(-.12); if(control.dataset.graphAction==='reset')state.graphController.reset();
  });

  window.addEventListener('hashchange', routeFromLocation);
  if (state.theme === 'light') document.body.classList.add('light-theme');
  renderNotifications();
  routeFromLocation();
  updateDataStatus();
})();
