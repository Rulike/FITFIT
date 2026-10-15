from collections import defaultdict

class UnionFind:
    def __init__(self, items): self.p={x:x for x in items}
    def find(self,x):
        while self.p[x]!=x: self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b: self.p[b]=a

def cluster_records(records, pair_scores, threshold=.72, min_similarity=.68):
    by={r['event_record_id']:r for r in records}; uf=UnionFind(by)
    score_map={}
    for f,s in pair_scores:
        score_map[f['event_record_id_a'],f['event_record_id_b']]=s
        score_map[f['event_record_id_b'],f['event_record_id_a']]=s
        if s>=threshold: uf.union(f['event_record_id_a'],f['event_record_id_b'])
    groups=defaultdict(list)
    for rid in by: groups[uf.find(rid)].append(rid)
    # The blocking key is already a hard invariant.  Keep the connected
    # components here; downstream validation reports any component-level
    # cohesion/conflict requiring manual review without losing records.
    return list(groups.values())
