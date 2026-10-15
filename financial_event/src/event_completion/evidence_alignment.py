def align_field(value, records):
    return [r.get('article_id') for r in records if value and str(value) in str(r)]
