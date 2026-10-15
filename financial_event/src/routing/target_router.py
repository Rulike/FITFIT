#!/usr/bin/env python3
"""target_router.py — 基于正式名称和唯一别名的目标路由器"""

import json, re, os
from collections import defaultdict
import unicodedata

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TargetRouter:
    def __init__(self, schema_file=None, alias_file=None, industry_keywords_file=None):
        if schema_file is None:
            schema_file = os.path.join(PROJECT_ROOT, "data/processed/event_schema.jsonl")
        if alias_file is None:
            alias_file = os.path.join(PROJECT_ROOT, "configs/target_aliases.json")

        # Load schemas
        schemas = []
        with open(schema_file) as f:
            for line in f:
                line = line.strip()
                if line: schemas.append(json.loads(line))

        self.schema_map = {s["schema_id"]: s for s in schemas}
        self.schemas_by_target = defaultdict(list)
        self.targets = {}  # (type, code) -> name
        for s in schemas:
            key = (s["target_type"], str(s["target_code"]))
            self.schemas_by_target[key].append(s)
            self.targets[key] = s["target_name"]

        # Build name lookup: normalized_name -> list of (type, code, name, is_formal)
        self.name_lookup = defaultdict(list)
        for (tt, tc), tn in self.targets.items():
            self.name_lookup[self._norm(tn)].append((tt, tc, tn, True))  # formal

        # Load aliases
        with open(alias_file) as f:
            aliases = json.load(f)
        for tt in ["company", "industry"]:
            for tn, info in aliases.get(tt, {}).items():
                for a in info.get("aliases", []):
                    na = self._norm(a)
                    if len(self.name_lookup[na]) == 0:
                        self.name_lookup[na].append((tt, info["code"], tn, False))

        # Build keyword index for industry routing
        self.industry_keywords = defaultdict(set)  # keyword -> set of (type, code)
        for (tt, tc), slist in self.schemas_by_target.items():
            if tt != "industry": continue
            tn = self.targets[(tt, tc)]
            # Extract keywords from event names and descriptions
            for s in slist:
                words = self._extract_keywords(s.get("event_name", "") + " " + s.get("raw_description", ""))
                for w in words:
                    if len(w) >= 2 and not self._is_generic(w):
                        self.industry_keywords[w].add((tt, tc))

    def _norm(self, text):
        if not text: return ""
        text = unicodedata.normalize('NFKC', text)
        # Full-width to half-width
        result = []
        for ch in text:
            code = ord(ch)
            if 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            elif code == 0x3000:
                result.append(' ')
            else:
                result.append(ch)
        text = ''.join(result)
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def _extract_keywords(self, text):
        """Extract Chinese word bigrams from text."""
        chinese = re.findall(r'[一-鿿]{2,}', text)
        return chinese

    def _is_generic(self, word):
        generic = {"行业","政策","事件","定义","示例","公司","企业","市场","发展","影响","变化","新闻","其中","包括","以及","进行","通过","使用","可以","能够","需要","应该","已经","正在","可能","相关","主要","重要","关键","部分","方面","情况","问题","方式","方法","过程","结果","作用","关系","因素","特点","特征","趋势","方向","目标","任务","重点","核心","基础","基本","总体","综合","整体","具体","实际","一般","一定","不同","相同","类似","直接","间接","明显","显著","突出","重要","重大","新的","新的","主要","当前","目前","以后","之前","以来","以后","一些","多种","大量","少数","某些","某个","包括","关于","对于","根据","按照","通过","为了","由于","因为","所以","然后","首先","其次","最后","同时","此外","另外","特别","尤其","更加","越来越","不断","持续","仍然","仍然","依然","继续","开始","结束","完成","实现","达到","超过","降低","提高","增加","减少","扩大","缩小","推动","促进","加强","加快","加速","提升","改善","优化","调整","转变","转型","升级","创新","突破","发展","增长","上升","下降","波动","变化"}
        return word in generic or len(word) < 2

    def route(self, title, content):
        """Route a single article to target entities. Returns list of routed targets."""
        title_norm = self._norm(title) if title else ""
        content_norm = self._norm(content) if content else ""
        full_norm = title_norm + " " + (content_norm[:5000] if content_norm else "")

        results = []
        seen = set()

        # 1. Title exact formal name match (priority: formal > alias)
        for name_key, matches in self.name_lookup.items():
            if len(name_key) < 3: continue
            if name_key in title_norm:
                for (tt, tc, tn, is_formal) in matches:
                    key = (tt, tc)
                    if key not in seen:
                        conf = 1.0 if is_formal else 0.95
                        idx = title_norm.find(name_key)
                        results.append({
                            "target_type": tt, "target_code": tc, "target_name": tn,
                            "route_method": "title_exact_target_name" if is_formal else "title_exact_unique_alias",
                            "route_confidence": conf, "matched_alias": name_key,
                            "matched_text_scope": "title",
                            "match_start": idx, "match_end": idx + len(name_key),
                        })
                        seen.add(key)

        # 2. Content match — only for industry names (company requires title for precision)
        for name_key, matches in self.name_lookup.items():
            if len(name_key) < 3: continue
            if name_key in content_norm:
                for (tt, tc, tn, is_formal) in matches:
                    if tt == "company": continue  # Company content match disabled for precision
                    key = (tt, tc)
                    if key not in seen:
                        conf = 0.80 if is_formal else 0.75
                        idx = content_norm.find(name_key)
                        results.append({
                            "target_type": tt, "target_code": tc, "target_name": tn,
                            "route_method": "content_industry_name",
                            "route_confidence": conf, "matched_alias": name_key,
                            "matched_text_scope": "content",
                            "match_start": idx, "match_end": idx + len(name_key),
                        })
                        seen.add(key)

        # 3. Content match for industry names only (conservative, title was already done)
        for name_key, matches in self.name_lookup.items():
            if len(name_key) < 3: continue
            if name_key in content_norm:
                for (tt, tc, tn, is_formal) in matches:
                    if tt != "industry": continue  # Only industry gets content match; company requires title
                    key = (tt, tc)
                    if key not in seen:
                        conf = 0.80 if is_formal else 0.75
                        idx = content_norm.find(name_key)
                        results.append({
                            "target_type": tt, "target_code": tc, "target_name": tn,
                            "route_method": "content_industry_name",
                            "route_confidence": conf, "matched_alias": name_key,
                            "matched_text_scope": "content",
                            "match_start": idx, "match_end": idx + len(name_key),
                        })
                        seen.add(key)

        return results
