#!/usr/bin/env python3
"""build_target_aliases.py — 构建保守目标别名字典"""

import json, os, re, sys
from collections import defaultdict
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_FILE = os.path.join(PROJECT_ROOT, "data/processed/event_schema.jsonl")
TRAIN_XLSX = os.path.join(PROJECT_ROOT, "data/raw/news_with_tags_train.xlsx")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

DANGEROUS_WORDS = {"科技","集团","股份","汽车","证券","能源","电信","联通","银行","投资","建设","传媒","影视","农业","钢铁","光伏","电池","医药","通信","电力","石油","电子","地产","保险","基金","信托","期货","租赁","水务","燃气","环保"}

def normalize(text):
    if not text: return ""
    text = text.strip()
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
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_aliases(name, target_type):
    aliases = set()
    n = normalize(name)
    aliases.add(n)  # Original normalized

    # Remove suffixes
    suffixes = ["股份有限公司","有限责任公司","有限公司","股份","（上海）","（深圳）","(上海)","(深圳)"]
    for suf in suffixes:
        if n.endswith(suf):
            short = n[:-len(suf)].strip()
            if len(short) >= 2: aliases.add(short)

    # Lowercase English
    lower = n.lower()
    if lower != n: aliases.add(lower)

    # ST variants
    if n.startswith("*ST") or n.startswith("ST"):
        aliases.add(n.lstrip("*").strip())

    return list(aliases)

def main():
    os.makedirs(CONFIG_DIR, exist_ok=True); os.makedirs(REPORTS_DIR, exist_ok=True)

    schemas = []
    with open(SCHEMA_FILE) as f:
        for line in f:
            line = line.strip()
            if line: schemas.append(json.loads(line))

    # Build target list
    targets = {}
    for s in schemas:
        key = (s["target_type"], str(s["target_code"]))
        if key not in targets:
            targets[key] = s["target_name"]

    # Generate aliases
    all_aliases = defaultdict(list)  # alias -> [(type, code, name)]
    target_alias_map = {}  # (type, code) -> [aliases]

    for (tt, tc), tn in targets.items():
        aliases = generate_aliases(tn, tt)
        target_alias_map[(tt, tc)] = aliases
        for a in aliases:
            all_aliases[a].append((tt, tc, tn))

    # Filter dangerous + ambiguous
    valid_aliases = {}
    ambiguous = []
    for alias, matches in all_aliases.items():
        if len(matches) > 1:
            ambiguous.append({
                "alias": alias, "target_type": matches[0][0],
                "matched_target_codes": ",".join(m[1] for m in matches),
                "matched_target_names": ",".join(m[2] for m in matches),
                "reason": "ambiguous_multiple_targets"
            })
            continue
        if alias in DANGEROUS_WORDS:
            continue
        if len(alias) < 3 and any('一' <= c <= '鿿' for c in alias):
            continue
        if len(alias) < 4 and not any('一' <= c <= '鿿' for c in alias):
            continue
        tt, tc, tn = matches[0]
        valid_aliases[(tt, tc)] = valid_aliases.get((tt, tc), []) + [alias]

    # Build output dict: target_name -> aliases
    alias_dict = {"company": {}, "industry": {}}
    for (tt, tc), aliases in valid_aliases.items():
        tn = targets[(tt, tc)]
        unique_aliases = sorted(set(a for a in aliases if a != tn))
        alias_dict[tt][tn] = {"code": tc, "aliases": unique_aliases}

    # Save
    with open(os.path.join(CONFIG_DIR, "target_aliases.json"), "w", encoding="utf-8") as f:
        json.dump(alias_dict, f, ensure_ascii=False, indent=2)

    pd.DataFrame(ambiguous).to_csv(os.path.join(REPORTS_DIR, "ambiguous_target_aliases.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(CONFIG_DIR, "manual_target_aliases.json"), "w", encoding="utf-8") as f:
        json.dump({"company": {}, "industry": {}}, f, ensure_ascii=False, indent=2)

    total_aliases = sum(len(v["aliases"]) for tt in alias_dict for v in alias_dict[tt].values())
    total_targets = len(companies := [k for k in targets if k[0]=="company"])
    total_industries = len([k for k in targets if k[0]=="industry"])

    summary = {
        "company_target_count": total_targets,
        "industry_target_count": total_industries,
        "total_target_count": len(targets),
        "generated_safe_alias_count": total_aliases,
        "ambiguous_alias_count": len(ambiguous),
        "dangerous_words_blocked": len(DANGEROUS_WORDS),
    }
    with open(os.path.join(REPORTS_DIR, "target_alias_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Also save inventory
    multi = sum(1 for (tt,tc), slist in defaultdict(list).items() if tt=="company")
    inventory = {
        "company_target_count": total_targets,
        "industry_target_count": total_industries,
        "total_target_count": len(targets),
        "schema_count": len(schemas),
        "targets_with_empty_schema": [],
        "targets_with_single_schema": sum(1 for (tt,tc) in targets if len([s for s in schemas if s["target_type"]==tt and str(s["target_code"])==tc]) == 1),
        "targets_with_multiple_schemas": sum(1 for (tt,tc) in targets if len([s for s in schemas if s["target_type"]==tt and str(s["target_code"])==tc]) > 1),
    }
    with open(os.path.join(REPORTS_DIR, "target_schema_inventory.json"), "w") as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2)

    print(f"Targets: {total_targets} companies, {total_industries} industries")
    print(f"Safe aliases: {total_aliases}, Ambiguous: {len(ambiguous)}")
    print(f"Single-schema: {inventory['targets_with_single_schema']}, Multi-schema: {inventory['targets_with_multiple_schemas']}")
    print("Done!")

if __name__ == "__main__":
    main()
