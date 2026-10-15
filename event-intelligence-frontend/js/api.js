/**
 * FinSight 数据接口层
 * --------------------------------------------------------------------------
 * 默认直接读取本地后端 `/api/v1`。
 * 需要切换 mock 时，把 USE_MOCK 设为 true 即可。
 */
(function () {
  'use strict';

  const CONFIG = {
    USE_MOCK: false,
    BASE_URL: '/api/v1',
    TIMEOUT: 15000,
    RETRIES: 1
  };

  const inflightGets = new Map();

  const wait = (value, delay = 120) => new Promise((resolve) => {
    window.setTimeout(() => resolve(structuredCloneSafe(value)), delay);
  });

  function structuredCloneSafe(value) {
    return typeof structuredClone === 'function'
      ? structuredClone(value)
      : JSON.parse(JSON.stringify(value));
  }

  async function requestOnce(path, options, attempt) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), CONFIG.TIMEOUT * (attempt + 1));
    try {
      const response = await fetch(`${CONFIG.BASE_URL}${path}`, {
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
        signal: controller.signal,
        ...options
      });
      if (!response.ok) {
        let message = `接口请求失败：${response.status}`;
        try {
          const payload = await response.json();
          message = payload.message || payload.error || message;
        } catch (_) { /* Non-JSON error response. */ }
        const error = new Error(message);
        error.status = response.status;
        throw error;
      }
      return await response.json();
    } finally {
      window.clearTimeout(timer);
    }
  }

  function isTransientError(error) {
    return error?.name === 'AbortError'
      || error instanceof TypeError
      || Number(error?.status) >= 500;
  }

  function request(path, options = {}) {
    const method = String(options.method || 'GET').toUpperCase();
    const requestKey = method === 'GET' ? `${method}:${path}` : '';
    if (requestKey && inflightGets.has(requestKey)) return inflightGets.get(requestKey);

    const task = (async () => {
      for (let attempt = 0; attempt <= CONFIG.RETRIES; attempt += 1) {
        try {
          return await requestOnce(path, options, attempt);
        } catch (error) {
          if (attempt >= CONFIG.RETRIES || method !== 'GET' || !isTransientError(error)) throw error;
          await new Promise((resolve) => window.setTimeout(resolve, 250 * (attempt + 1)));
        }
      }
      throw new Error('接口请求失败');
    })();

    if (!requestKey) return task;
    inflightGets.set(requestKey, task);
    task.finally(() => {
      if (inflightGets.get(requestKey) === task) inflightGets.delete(requestKey);
    }).catch(() => {});
    return task;
  }

  const mock = window.MockData || {};

  const defaultRules = { high: 75, critical: 90, credibility: 0, channel: '平台消息' };
  const mockAcknowledged = new Set();

  function mockEventResult(id, build) {
    const event = (mock.events || []).find(item => item.id === id);
    return event ? wait(build(event)) : Promise.reject(new Error('该事件不存在，请重新选择'));
  }

  function mockAlerts() {
    const rules = mock.alertRules || defaultRules;
    return (mock.events || []).filter(event => event.impact >= rules.high && event.credibility >= rules.credibility)
      .sort((a, b) => b.impact - a.impact)
      .map(event => {
        const level = event.impact >= rules.critical ? 'critical' : 'high';
        return { id: `ALT-${event.id}`, eventId: event.id, title: event.title, company: `${event.company} · ${event.code}`,
          reason: `${event.type} · 影响分 ${event.impact}`, time: event.time, score: event.impact, level,
          status: mockAcknowledged.has(event.id) ? '跟踪中' : '待处置',
          color: level === 'critical' ? '#ff5d73' : '#ffb648', icon: level === 'critical' ? '!' : '◇' };
      });
  }

  function readUsers() {
    try { return JSON.parse(localStorage.getItem('finsight-users') || '[]'); }
    catch (_) { return []; }
  }

  function writeUsers(users) {
    try { localStorage.setItem('finsight-users', JSON.stringify(users)); }
    catch (_) { /* Local-file privacy mode may disable storage. */ }
  }

  window.EventAPI = {
    config: CONFIG,
    getDashboard: () => CONFIG.USE_MOCK ? wait(mock.dashboard) : request('/dashboard'),
    getEvents: (filters = {}) => CONFIG.USE_MOCK
      ? wait(mock.events.filter((event) => {
          const query = (filters.query || '').toLowerCase();
          return (!query || `${event.title}${event.company}${event.code}`.toLowerCase().includes(query))
            && (!filters.level || filters.level === 'all' || event.level === filters.level)
            && (!filters.sentiment || filters.sentiment === 'all' || event.sentiment === filters.sentiment);
        }))
      : request(`/events?${new URLSearchParams(filters)}`),
    getEventDetail: (id) => CONFIG.USE_MOCK
      ? (mock.events.some(event => event.id === id) ? wait(mock.events.find(event => event.id === id)) : Promise.reject(new Error('该事件不存在，请重新选择')))
      : request(`/events/${encodeURIComponent(id || '')}`),
    getCredibility: (id = 'EVT-240816-001') => CONFIG.USE_MOCK
      ? mockEventResult(id, event => ({ eventId: id, score: event.credibility, verdict: '离线演示样本', summary: event.summary, evidence: [], reasons: [{ label: '样本说明', value: '仅供界面演示', note: '未连接项目证据接口' }] }))
      : request(`/events/${encodeURIComponent(id)}/credibility`),
    getTimeline: (id = 'EVT-240816-001') => CONFIG.USE_MOCK
      ? mockEventResult(id, event => ({ eventId: id, title: event.title, company: event.company, code: event.code, impact: event.impact,
          stages: [{ name: '已有样本', time: event.time, state: 'done' }],
          items: [{ date: event.time, time: '', title: event.title, text: event.summary, source: event.source, type: '演示样本', color: '#4aa8ff' }],
          impacts: [{ name: '影响方向', value: event.direction }, { name: '样本状态', value: event.status }] }))
      : request(`/events/${encodeURIComponent(id)}/timeline`),
    getGraph: (scope = 'company') => CONFIG.USE_MOCK
      ? wait({ ...mock.graph, scope })
      : request(`/knowledge-graph?scope=${encodeURIComponent(scope)}`),
    getDataStatus: () => CONFIG.USE_MOCK ? wait({}) : request('/status'),
    getAlertRules: () => CONFIG.USE_MOCK ? wait(mock.alertRules || defaultRules) : request('/alert-rules'),
    saveAlertRules: (payload) => {
      if (!CONFIG.USE_MOCK) return request('/alert-rules', { method: 'POST', body: JSON.stringify(payload) });
      const values = ['high', 'critical', 'credibility'].map(key => Number(payload[key]));
      if (values.some(value => !Number.isFinite(value) || value < 0 || value > 100) || values[0] >= values[1]) return Promise.reject(new Error('阈值应为 0–100，且重大阈值高于高影响阈值'));
      mock.alertRules = { ...defaultRules, high: values[0], critical: values[1], credibility: values[2] };
      return wait(mock.alertRules);
    },
    getWatches: () => CONFIG.USE_MOCK ? wait(mock.watches || []) : request('/watches'),
    getAlerts: () => CONFIG.USE_MOCK ? wait(mockAlerts()) : request('/alerts'),
    acknowledgeAlert: (id) => {
      if (!CONFIG.USE_MOCK) return request(`/alerts/${encodeURIComponent(id)}/acknowledge`, { method: 'POST' });
      const alert = mockAlerts().find((item) => item.id === id);
      if (!alert) return Promise.reject(new Error('该预警不存在，请刷新后重试'));
      mockAcknowledged.add(alert.eventId);
      return wait({ success: true, id }, 260);
    },
    getCompanyProfile: (code = '300750') => CONFIG.USE_MOCK
      ? wait({ ...mock.company, code })
      : request(`/companies/${encodeURIComponent(code)}`),
    getModelMetrics: () => CONFIG.USE_MOCK ? wait(mock.models) : request('/models/metrics'),
    getWorkbench: () => CONFIG.USE_MOCK ? wait(mock.workbench || {}) : request('/workbench'),
    getProcessDemo: (limit = 5) => CONFIG.USE_MOCK
      ? wait(mock.processDemo || {})
      : request(`/process-demo?${new URLSearchParams({ limit })}`),
    createWatch: (payload) => CONFIG.USE_MOCK
      ? (() => { const watch = { ...payload, id: `WATCH-${Date.now()}` }; (mock.watches ||= []).push(watch); return wait({ success: true, ...watch }); })()
      : request('/watches', { method: 'POST', body: JSON.stringify(payload) }),
    search: (query) => {
      if (!CONFIG.USE_MOCK) return request(`/search?q=${encodeURIComponent(query)}`);
      const key = query.trim().toLowerCase();
      const results = mock.events
        .filter((item) => !key || `${item.company}${item.code}${item.title}${item.type}`.toLowerCase().includes(key))
        .slice(0, 6)
        .map((item) => ({ type: 'event', id: item.id, title: item.title, meta: `${item.company} · ${item.code}`, level: item.level }));
      return wait(results, 60);
    },
    login: (payload) => {
      if (!CONFIG.USE_MOCK) return request('/auth/login', { method: 'POST', body: JSON.stringify(payload) });
      const users = readUsers();
      const isDemo = payload.account === 'demo@finsight.cn' && payload.password === '123456';
      const user = users.find((item) => item.account === payload.account && item.password === payload.password);
      if (!isDemo && !user) return Promise.reject(new Error('账号或密码不正确'));
      return wait({ success: true, token: `demo-token-${Date.now()}`, user: isDemo ? { name: '演示用户', account: payload.account } : { name: user.name, account: user.account } }, 420);
    },
    register: (payload) => {
      if (!CONFIG.USE_MOCK) return request('/auth/register', { method: 'POST', body: JSON.stringify(payload) });
      const users = readUsers();
      if (users.some((item) => item.account === payload.account) || payload.account === 'demo@finsight.cn') return Promise.reject(new Error('该账号已注册'));
      users.push({ name: payload.name, account: payload.account, password: payload.password });
      writeUsers(users);
      return wait({ success: true, user: { name: payload.name, account: payload.account } }, 520);
    }
  };
})();
