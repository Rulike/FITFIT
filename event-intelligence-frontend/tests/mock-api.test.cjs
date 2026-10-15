const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
function setup() {
  const context = vm.createContext({ window: {}, setTimeout, clearTimeout, URLSearchParams, structuredClone });
  context.window.setTimeout = setTimeout;
  for (const file of ['js/mock-data.js', 'js/api.js']) vm.runInContext(fs.readFileSync(file, 'utf8'), context);
  context.window.EventAPI.config.USE_MOCK = true;
  return context.window.EventAPI;
}
test('mock alerts refer to real mock events and apply saved rules', async () => {
  const api = setup();
  const alerts = await api.getAlerts();
  for (const alert of alerts) assert.equal((await api.getEventDetail(alert.eventId)).id, alert.eventId);
  await api.saveAlertRules({ high: 90, critical: 95, credibility: 0 });
  const filtered = await api.getAlerts();
  assert.ok(filtered.length < alerts.length);
  assert.ok(filtered.every(item => item.score >= 90 && item.level === 'high'));
  await api.acknowledgeAlert(filtered[0].id);
  assert.equal((await api.getAlerts()).find(item => item.id === filtered[0].id).status, '跟踪中');
});
test('mock timeline uses the requested event', async () => {
  const api = setup();
  const events = await api.getEvents();
  const timeline = await api.getTimeline(events[1].id);
  assert.equal(timeline.title, events[1].title);
  assert.equal(timeline.company, events[1].company);
});
test('invalid mock rule leaves settings unchanged', async () => {
  const api = setup();
  await assert.rejects(api.saveAlertRules({ high: 99, critical: 80, credibility: 0 }));
  assert.equal((await api.getAlertRules()).high, 75);
});
