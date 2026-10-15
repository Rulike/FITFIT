def build_prompt(cluster, records, conflicts):
    return {'schema':cluster.get('schema_id',''),'target':cluster.get('target_name',''),'articles':[{'article_id':r.get('article_id'),'title':r.get('title'),'evidence_sentences':r.get('evidence_sentences',[]),'fields':r} for r in records],'rule_merge':cluster,'conflicts':conflicts}
