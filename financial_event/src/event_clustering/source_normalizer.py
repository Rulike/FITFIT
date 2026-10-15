import json,re,unicodedata
from pathlib import Path

FORBIDDEN={'quality_status','final_status','route_method','confidence_level','auto_accept','llm_accepted','joint_adjudication','manual_review','reject','rejected','high','medium','low'}
def normalize_source(value, aliases=None):
    raw='' if value is None else str(value)
    cleaned=unicodedata.normalize('NFKC',raw).strip()
    cleaned=re.sub(r'\s+',' ',cleaned)
    if not cleaned or cleaned.lower() in FORBIDDEN: return 'unknown'
    aliases=aliases or {}
    return aliases.get(cleaned,aliases.get(cleaned.lower(),cleaned)) or 'unknown'

def is_invalid_source(value):
    return normalize_source(value) == 'unknown'
def load_aliases(path):
    p=Path(path)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
