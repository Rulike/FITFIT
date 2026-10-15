import re
from datetime import datetime

def parse_dt(value):
    if not value: return None
    s = str(value).strip().replace('T',' ')
    try: return datetime.fromisoformat(s[:19])
    except ValueError:
        m = re.search(r'(20\d{2})[-年](\d{1,2})[-月](\d{1,2})', s)
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None

def event_date(row):
    et = row.get('event_time')
    if isinstance(et, dict): et = et.get('normalized_date') or et.get('date')
    return parse_dt(et) or parse_dt(row.get('publish_time'))

def time_window_days(row):
    name = f"{row.get('event_name','')} {row.get('schema_id','')}".lower()
    persistent = any(x in name for x in ('政策','产能','诉讼','项目','进展','研发','供应链'))
    return 90 if persistent else (30 if row.get('target_type') == 'company' else 15)

def blocking_key(row):
    return (row.get('target_type',''), row.get('target_code',''), row.get('schema_id',''))

def token_set(value):
    text = str(value or '').lower()
    return set(re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', text))

def jaccard(a,b):
    a,b=token_set(a),token_set(b)
    return len(a&b)/len(a|b) if a|b else 0.0
