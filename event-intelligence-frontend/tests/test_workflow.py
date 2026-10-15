import importlib
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import api_server as api


def event(identifier, impact=95, credibility=82):
    return dict(id=identifier, title=identifier, company='测试公司', code='000001',
                level='critical' if impact >= 90 else 'high', impact=impact,
                sourceCount=2, credibility=credibility, sentiment='negative',
                type='业绩变化', time='09-16', relatedCount=3)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_workflow = getattr(api, 'WORKFLOW', None)
        if self.old_workflow is not None:
            module = importlib.import_module('workflow_store')
            api.WORKFLOW = module.WorkflowStore(Path(self.temp.name) / 'workflow.sqlite3')
        self.store = api.DataStore.__new__(api.DataStore)
        self.store.events = [event('one'), event('two', 85)]
        self.store.event_by_id = {item['id']: item for item in self.store.events}
        self.store.company_counter = Counter({'测试公司': 2})
        self.old_store = api.STORE
        api.STORE = self.store

    def tearDown(self):
        api.STORE = self.old_store
        if self.old_workflow is not None:
            api.WORKFLOW = self.old_workflow
        self.temp.cleanup()

    def request(self, path, payload=None, method='POST'):
        handler = api.Handler.__new__(api.Handler)
        result = {}
        handler.read_body = lambda: payload or {}
        handler.json_response = lambda body, status=200: result.update(body=body, status=status)
        if method == 'POST':
            handler.handle_api_post(path)
        else:
            handler.handle_api_get(path, {})
        return result

    def test_alert_id_survives_new_higher_ranked_event(self):
        before = {item['eventId']: item['id'] for item in self.store.alerts()}
        self.store.events.insert(0, event('new', 100))
        after = {item['eventId']: item['id'] for item in self.store.alerts()}
        self.assertEqual(before['one'], after['one'])

    def test_acknowledgement_persists_after_reopening_store(self):
        alert = self.store.alerts()[0]
        result = self.request('/alerts/' + alert['id'] + '/acknowledge')
        self.assertEqual(result['status'], 200)
        self.assertEqual(self.store.alerts()[0]['status'], '跟踪中')
        module = importlib.import_module('workflow_store')
        api.WORKFLOW = module.WorkflowStore(Path(self.temp.name) / 'workflow.sqlite3')
        self.assertEqual(self.store.alerts()[0]['status'], '跟踪中')

    def test_unknown_acknowledgement_is_rejected(self):
        self.assertEqual(self.request('/alerts/missing/acknowledge')['status'], 404)

    def test_rule_saved_and_applied(self):
        result = self.request('/alert-rules', dict(high=90, critical=98, credibility=80))
        self.assertEqual(result['status'], 200)
        alerts = self.store.alerts()
        self.assertEqual([item['eventId'] for item in alerts], ['one'])
        self.assertEqual(alerts[0]['level'], 'high')
        self.assertEqual(self.request('/alert-rules', method='GET')['body']['high'], 90)

    def test_invalid_rules_leave_previous_rule_unchanged(self):
        self.assertEqual(self.request('/alert-rules', dict(high=99, critical=90, credibility=0))['status'], 400)
        self.assertEqual(self.request('/alert-rules', method='GET')['body']['high'], 75)

    def test_unknown_event_never_returns_first_event(self):
        for suffix in ['', '/credibility', '/timeline']:
            self.assertEqual(self.request('/events/missing' + suffix, method='GET')['status'], 404)

    def test_graph_node_counts_are_from_actual_edges(self):
        graph = self.store.graph()
        for node in graph['nodes']:
            neighbors = {b if a == node['id'] else a for a, b, *_ in graph['links'] if node['id'] in (a, b)}
            self.assertEqual(node.get('relations'), len(neighbors))
        company = next(node for node in graph['nodes'] if node['type'] == 'company')
        self.assertEqual(company.get('events'), 2)
        self.assertIsNone(company.get('credibility'))

    def test_watch_is_saved_and_can_be_read(self):
        result = self.request('/watches', dict(target='测试公司', scope='上市公司事件', level='高影响及以上', keywords='业绩'))
        self.assertEqual(result['status'], 200)
        watches = self.request('/watches', method='GET')['body']
        self.assertEqual(watches[0]['target'], '测试公司')
        self.assertEqual(watches[0]['id'], result['body']['id'])

    def test_concurrent_acknowledgements_do_not_lose_updates(self):
        module = importlib.import_module('workflow_store')
        store = module.WorkflowStore(Path(self.temp.name) / 'parallel.sqlite3')
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(store.acknowledge, ['a', 'b', 'c', 'd']))
        self.assertEqual(set(store.acknowledgements()), {'a', 'b', 'c', 'd'})


if __name__ == '__main__':
    unittest.main()
