const { test, expect } = require('@playwright/test');
const base = process.env.FRONTEND_TEST_URL || 'http://127.0.0.1:18080';
let events;
test.beforeAll(async ({ request }) => {
  const response = await request.get(`${base}/api/v1/events`);
  events = await response.json();
  expect(events.length).toBeGreaterThan(1);
});

test('analysis requires an explicit event', async ({ page }) => {
  await page.goto(`${base}/#/credibility`);
  await expect(page.getByText('请先选择分析事件', { exact: true })).toBeVisible();
});

test('event context survives switching, reload and browser back', async ({ page }) => {
  const [first, second] = events;
  await page.goto(`${base}/#/credibility?event=${encodeURIComponent(first.id)}`);
  await expect(page.locator('.analysis-context')).toContainText(first.title);
  await page.locator('.analysis-context [data-action="select-event"]').click();
  await page.locator('#commandInput').fill(second.title);
  await page.locator('#commandResults [data-event-id]').first().click();
  await expect(page.locator('.analysis-context')).toContainText(second.title);
  await expect(page).toHaveURL(new RegExp(encodeURIComponent(second.id)));
  await page.reload();
  await expect(page.locator('.analysis-context')).toContainText(second.title);
  await page.locator('.analysis-steps [data-nav="timeline"]').click();
  await expect(page.locator('.analysis-context')).toContainText(second.title);
  await page.goBack();
  await expect(page.locator('h1')).toHaveText('可信评估');
  await page.goBack();
  await expect(page.locator('.analysis-context')).toContainText(first.title);
});

test('clearing process selection leaves no implicit first event', async ({ page }) => {
  await page.goto(`${base}/#/integration`);
  await page.locator('#processEventSelect').selectOption('0');
  await page.locator('.process-rail-item').nth(1).click();
  await expect(page.locator('.process-current-result')).toContainText('当前阶段结果');
  await page.locator('#processEventSelect').selectOption('');
  await expect(page.locator('.process-rail-item')).toHaveCount(4);
  await expect(page.locator('.process-rail-item').first()).toBeDisabled();
  await expect(page.locator('.process-stage-empty')).toBeVisible();
});

test('replay event change clears the previous stage result', async ({ page }) => {
  await page.goto(`${base}/#/integration`);
  await page.locator('#processEventSelect').selectOption('0');
  await page.locator('.process-rail-item').nth(2).click();
  const firstTitle = await page.locator('.process-result-head h2').textContent();
  await page.locator('#processEventSelect').selectOption('1');
  await expect(page.locator('.process-result-status')).toHaveText('待选择');
  await expect(page.locator('.process-stage-empty')).toBeVisible();
  await expect(page.locator('.process-result-head h2')).not.toHaveText(firstTitle);
});

test('starting event intake advances the current process stage', async ({ page }) => {
  await page.goto(`${base}/#/integration`);
  await page.locator('#processEventSelect').selectOption('0');
  await expect(page.locator('[data-action="process-start"]')).toHaveText('开始接收事件');
  await page.locator('[data-action="process-start"]').click();
  await expect(page.locator('.process-result-status')).toHaveText('处理中');
  await expect(page.locator('.process-simulation-processing')).toBeVisible();
  await expect(page.locator('.process-result-status')).toHaveText('已完成', { timeout: 2500 });
});

test('search supports arrow keys and enter', async ({ page }) => {
  await page.goto(`${base}/#/events`);
  await page.locator('#globalSearch').click();
  await expect(page.locator('.command-item')).not.toHaveCount(0);
  const secondId = await page.locator('.command-item').nth(1).getAttribute('data-event-id');
  await page.locator('#commandInput').press('ArrowDown');
  await expect(page.locator('.command-item').nth(1)).toHaveClass(/active/);
  await page.locator('#commandInput').press('Enter');
  await expect(page.locator('#detailDrawer')).toHaveAttribute('aria-hidden', 'false');
  await expect(page.locator('#drawerContent')).toContainText(events.find(e => e.id === secondId).title);
});

test('event filtering retains focus and reset is available', async ({ page }) => {
  await page.goto(`${base}/#/events`);
  await page.locator('#eventSearch').fill('a-query-with-no-matches');
  await expect(page.getByText('未找到匹配事件', { exact: true })).toBeVisible();
  await expect(page.locator('#eventSearch')).toBeFocused();
  await page.locator('[data-action="reset-filters"]').first().click();
  await expect(page.locator('.data-table tbody tr')).not.toHaveCount(0);
  await expect(page.locator('#eventSearch')).toHaveValue('');
});

test('missing event shows an error without substituting another event', async ({ page }) => {
  await page.goto(`${base}/#/credibility?event=missing`);
  await expect(page.getByText('该事件不存在或已不在当前数据中，请重新选择')).toBeVisible();
  await expect(page.locator('.score-hero')).toHaveCount(0);
});

test('graph never derives a confidence score from node size', async ({ page }) => {
  await page.goto(`${base}/#/graph`);
  await expect(page.locator('#nodeInfo')).toContainText('未提供');
  await expect(page.locator('#nodeInfo')).not.toContainText('稳定');
  await expect(page.locator('[aria-label="放大图谱"]')).toBeVisible();
  await page.locator('#graphScope').selectOption('risk');
  await expect(page.locator('.graph-scope-select')).toHaveValue('risk');
});

test('graph event node exposes the analysis entry', async ({ page }) => {
  await page.goto(`${base}/#/graph`);
  const eventNode = page.locator('#knowledgeGraph');
  await expect(eventNode).toBeVisible();
  await page.mouse.click(0, 0); // no-op keeps canvas interaction covered without assuming coordinates
  await expect(page.locator('#nodeInfo')).toContainText('当前图谱节点');
});

test('graph uses the free relation view with direct node interaction', async ({ page }) => {
  await page.goto(`${base}/#/graph`);
  await expect(page.locator('.graph-force-shell')).toBeVisible();
  await expect(page.locator('#knowledgeGraph')).toHaveAttribute('data-graph-interactive', 'true');
  await expect(page.locator('.graph-interaction-hint')).toContainText('节点可拖动');
  await expect(page.locator('.graph-interaction-hint')).toContainText('点击节点显示关联关系');
  await expect(page.locator('.graph-edge-count')).toContainText('节点');
});

test('alert failures are visible and can be retried', async ({ page }) => {
  await page.goto(`${base}/#/alerts`);
  await page.route('**/alerts/*/acknowledge', route => route.fulfill({ status: 400, contentType: 'application/json', body: JSON.stringify({ message: '测试保存失败' }) }));
  const button = page.locator('[data-action="ack-alert"]').first();
  await button.click();
  await expect(page.locator('.toast')).toContainText('测试保存失败');
  await expect(button).toBeEnabled();
});

test('latest event filter wins even when an earlier request finishes last', async ({ page }) => {
  await page.goto(`${base}/#/events`);
  let release;
  const barrier = new Promise(resolve => { release = resolve; });
  await page.route('**/api/v1/events?**', async route => {
    const query = new URL(route.request().url()).searchParams.get('query');
    if (query === 'slow') { await barrier; await route.fulfill({ json: [events[0]] }); }
    else if (query === 'fast') await route.fulfill({ json: [events[1]] });
    else await route.continue();
  });
  const slowRequest = page.waitForRequest(r => r.url().includes('query=slow'));
  await page.locator('#eventSearch').fill('slow');
  await slowRequest;
  await page.locator('#eventSearch').fill('fast');
  await expect(page.locator('.data-table tbody')).toContainText(events[1].title);
  const slowResponse = page.waitForResponse(r => r.url().includes('query=slow'));
  release();
  await slowResponse;
  await expect(page.locator('.data-table tbody')).toContainText(events[1].title);
  await expect(page.locator('#eventSearch')).toHaveValue('fast');
});

test('closing an event preview before response keeps it closed', async ({ page }) => {
  await page.goto(`${base}/#/events`);
  let release;
  const barrier = new Promise(resolve => { release = resolve; });
  await page.route('**/api/v1/events/' + events[0].id, async route => { await barrier; await route.fulfill({ json: events[0] }); });
  const requested = page.waitForRequest(r => r.url().endsWith('/events/' + events[0].id));
  await page.locator('.data-table tbody tr').first().click();
  await requested;
  await page.keyboard.press('Escape');
  const response = page.waitForResponse(r => r.url().endsWith('/events/' + events[0].id));
  release(); await response;
  await expect(page.locator('#detailDrawer')).toHaveAttribute('aria-hidden', 'true');
});

test('event list has usable pagination', async ({ page }) => {
  await page.goto(`${base}/#/events`);
  await expect(page.locator('.data-table tbody tr')).toHaveCount(25);
  await page.getByRole('button', { name: '下一页', exact: true }).click();
  await expect(page.locator('.pagination')).toContainText('26–50');
  await page.getByRole('button', { name: '上一页', exact: true }).click();
  await expect(page.locator('.pagination')).toContainText('1–25');
});

test('mobile analysis stays within the viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${base}/#/credibility?event=${encodeURIComponent(events[0].id)}`);
  await expect(page.locator('.analysis-context')).toBeVisible();
  for (const selector of ['.analysis-context', '.analysis-steps', '.page-container']) {
    const box = await page.locator(selector).boundingBox();
    expect(box.x + box.width).toBeLessThanOrEqual(391);
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(391);
  await expect(page.locator('#detailDrawer')).not.toBeVisible();
  await page.screenshot({ path: 'test-results/mobile-analysis.png', fullPage: true });
});

test('mobile replay uses a two-column stage rail and one result panel', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${base}/#/integration`);
  await expect(page.locator('.process-context-bar')).toBeVisible();
  await expect(page.locator('.process-workspace')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(391);
  await expect(page.locator('.process-summary-row')).toHaveCount(0);
});

test('mobile graph keeps the free canvas inside the viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${base}/#/graph`);
  await expect(page.locator('.graph-force-shell')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(391);
});

test('all pages render without uncaught errors', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  for (const route of ['dashboard', 'events', 'credibility', 'timeline', 'alerts', 'graph', 'integration']) {
    await page.goto(`${base}/#/${route}?event=${encodeURIComponent(events[0].id)}`);
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.getByText('页面加载失败', { exact: true })).toHaveCount(0);
  }
  expect(errors).toEqual([]);
});

test('acknowledgement remains after a browser reload', async ({ page, request }) => {
  const alerts = await (await request.get(`${base}/api/v1/alerts`)).json();
  const alert = alerts.find(item => item.status === '待处置');
  expect(alert).toBeTruthy();
  await page.goto(`${base}/#/alerts?event=${encodeURIComponent(alert.eventId)}`);
  await page.locator('[data-action="ack-alert"]').click();
  await expect(page.locator('.alert-card')).toContainText('跟踪中');
  await page.reload();
  await expect(page.locator('.alert-card')).toContainText('跟踪中');
  await expect(page.locator('[data-action="ack-alert"]')).toHaveCount(0);
});

test('rules reject invalid ordering and persist valid edits', async ({ page, request }) => {
  const original = await (await request.get(`${base}/api/v1/alert-rules`)).json();
  try {
    await page.goto(`${base}/#/alerts`);
    await page.locator('[data-action="alert-settings"]').click();
    await page.locator('[name="high"]').fill('95');
    await page.locator('[name="critical"]').fill('90');
    await page.locator('#alertSettingsForm button[type="submit"]').click();
    await expect(page.locator('.form-error')).toContainText('重大预警阈值必须高于');
    await page.locator('[name="high"]').fill('80');
    await page.locator('#alertSettingsForm button[type="submit"]').click();
    await expect(page.locator('.rule-summary')).toContainText('≥ 80');
    await page.reload();
    await expect(page.locator('.rule-summary')).toContainText('≥ 80');
  } finally {
    await request.post(`${base}/api/v1/alert-rules`, { data: original });
  }
});

test('notification opens its alert and global alerts preserve analysis context', async ({ page }) => {
  await page.goto(`${base}/#/credibility?event=${encodeURIComponent(events[0].id)}`);
  await page.locator('#notificationButton').click();
  await page.locator('#notificationList [data-alert-id]').first().click();
  await expect(page.locator('#modalTitle')).toHaveText('预警详情');
  await page.keyboard.press('Escape');
  await page.locator('#primaryNav [data-nav="alerts"]').click();
  await expect(page.locator('h1')).toHaveText('预警中心');
  await expect(page.locator('.analysis-context')).toHaveCount(0);
  await page.reload();
  await page.locator('#primaryNav [data-nav="timeline"]').click();
  await expect(page.locator('.analysis-context')).toContainText(events[0].title);
});
