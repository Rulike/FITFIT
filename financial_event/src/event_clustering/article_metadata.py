import pandas as pd
from .source_normalizer import normalize_source,load_aliases

def read_article_metadata(path, split, aliases=None):
    if not pd.io.common.file_exists(path):
        return []
    df=pd.read_excel(path,usecols=lambda c:c in {'article_file_id','article_title','article_source','article_publish_time'})
    if 'article_file_id' not in df.columns:
        raise ValueError(f'{path} must contain article_file_id')
    df['article_id']=df['article_file_id'].map(lambda x: '' if pd.isna(x) else str(x))
    rows=[]
    for r in df.itertuples(index=False):
        raw='' if pd.isna(r.article_source) else str(r.article_source)
        rows.append({'dataset_split':split,'article_id':str(r.article_id),
                     'article_title':'' if pd.isna(r.article_title) else str(r.article_title),
                     'article_source':raw,
                     'article_publish_time':'' if pd.isna(r.article_publish_time) else str(r.article_publish_time),
                     'source_normalized':normalize_source(raw,aliases)})
    return rows

def build_metadata_lookup(rows):
    grouped={}
    for row in rows:
        grouped.setdefault((str(row['dataset_split']),str(row['article_id'])),[]).append(row)
    return grouped
