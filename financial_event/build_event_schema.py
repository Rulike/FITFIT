#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_event_schema.py
---------------------
将公司事件体系（事件类型_标的.json）和行业事件体系（事件类型_行业.json）
转换为统一的事件 Schema 格式，输出 JSONL 和 Excel。

统一格式：
{
    "schema_id": "company_002594_新车出海",
    "target_type": "company",
    "target_code": "002594",
    "target_name": "比亚迪",
    "event_name": "新车出海",
    "raw_description": "完整的原始事件定义"
}
"""

import json
import os
import re
import sys
from collections import Counter

# ── 路径配置 ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

COMPANY_FILE = os.path.join(RAW_DIR, "事件类型_标的.json")
INDUSTRY_FILE = os.path.join(RAW_DIR, "事件类型_行业.json")

OUTPUT_JSONL = os.path.join(PROCESSED_DIR, "event_schema.jsonl")
OUTPUT_XLSX = os.path.join(PROCESSED_DIR, "event_schema.xlsx")
SUMMARY_JSON = os.path.join(REPORTS_DIR, "event_schema_summary.json")
FREQ_CSV = os.path.join(REPORTS_DIR, "event_name_frequency.csv")


# ── 工具函数 ──────────────────────────────────────────────
def clean_description(text: str) -> str:
    """
    清理事件描述文本：
    - 替换制表符为空格
    - 统一换行符为 \n
    - 去除每行首尾的多余空格
    - 将连续 3 个及以上空行合并为最多 1 个空行
    - 去除整段文本首尾空白，但保留内部段落结构
    """
    if not isinstance(text, str):
        return ""
    # 替换制表符
    text = text.replace("\t", " ")
    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 去除每行首尾空格
    lines = text.split("\n")
    lines = [line.strip() for line in lines]
    # 合并连续空行：将 >=3 个连续空行合并为 1 个空行
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    # 去除整段首尾空白
    text = text.strip()
    return text


def ensure_string(value) -> str:
    """确保值为字符串，保留前导零。"""
    if isinstance(value, (int, float)):
        # 对于整数型的浮点数（如 2594.0），还原为整数字符串
        if isinstance(value, float) and value == int(value):
            s = str(int(value))
        else:
            s = str(value)
        # 如果原始 JSON 中 code 可能以数字形式存储，尝试补齐到常见长度
        # 不做补齐，因为原始数据中 trading_code 就是字符串
        return s
    return str(value)


def make_schema_id(target_type: str, code: str, event_name: str) -> str:
    """生成唯一的 schema_id。"""
    return f"{target_type}_{code}_{event_name}"


# ── 主处理逻辑 ────────────────────────────────────────────
def load_json(filepath: str) -> list:
    """加载 JSON 文件。"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望顶层为数组，但得到 {type(data).__name__}: {filepath}")
    return data


def process_company_events(data: list) -> list:
    """处理公司事件，返回统一格式的记录列表。"""
    records = []
    for item in data:
        company = item.get("company", {})
        trading_code = ensure_string(company.get("trading_code", ""))
        secu_abbr = company.get("secu_abbr", "")
        event_schema = item.get("event_schema", [])

        for evt in event_schema:
            event_name = evt.get("event_name", "")
            description = clean_description(evt.get("description", ""))
            schema_id = make_schema_id("company", trading_code, event_name)
            records.append({
                "schema_id": schema_id,
                "target_type": "company",
                "target_code": trading_code,
                "target_name": secu_abbr,
                "event_name": event_name,
                "raw_description": description,
            })
    return records


def process_industry_events(data: list) -> list:
    """处理行业事件，返回统一格式的记录列表。"""
    records = []
    for item in data:
        industry = item.get("industry", {})
        inducode = ensure_string(industry.get("inducode", ""))
        induname = industry.get("induname", "")
        event_schema = item.get("event_schema", [])

        for evt in event_schema:
            event_name = evt.get("event_name", "")
            description = clean_description(evt.get("description", ""))
            schema_id = make_schema_id("industry", inducode, event_name)
            records.append({
                "schema_id": schema_id,
                "target_type": "industry",
                "target_code": inducode,
                "target_name": induname,
                "event_name": event_name,
                "raw_description": description,
            })
    return records


# ── 数据校验 ──────────────────────────────────────────────
def validate_records(records: list, label: str) -> list:
    """
    校验记录，返回异常信息列表。
    检查项：
    1. 必需字段是否存在
    2. 必需字段是否为空
    3. 代码是否为浮点数格式（不应出现 ".0" 等）
    """
    required_fields = ["schema_id", "target_type", "target_code", "target_name", "event_name", "raw_description"]
    errors = []

    for i, rec in enumerate(records):
        prefix = f"[{label} #{i}]"

        # 1. 检查必需字段是否存在
        for field in required_fields:
            if field not in rec:
                errors.append(f"{prefix} 缺少必需字段: {field}")

        # 2. 检查必需字段是否为空
        for field in required_fields:
            if field in rec and (rec[field] is None or (isinstance(rec[field], str) and rec[field].strip() == "")):
                errors.append(f"{prefix} 字段为空: {field}")

        # 3. 检查代码是否为浮点数格式
        code = rec.get("target_code", "")
        if isinstance(code, float):
            errors.append(f"{prefix} target_code 是浮点数类型: {code}")
        if isinstance(code, str) and re.search(r'\.0$', code):
            errors.append(f"{prefix} target_code 疑似浮点数转换: {code}")

        # 4. 检查 target_type
        tt = rec.get("target_type", "")
        if tt not in ("company", "industry"):
            errors.append(f"{prefix} target_type 值非法: {tt}")

    return errors


def check_duplicate_schema_ids(records: list) -> list:
    """检查重复的 schema_id，返回重复项列表。"""
    ids = [r["schema_id"] for r in records]
    counter = Counter(ids)
    duplicates = [sid for sid, count in counter.items() if count > 1]
    return duplicates


# ── 统计 ──────────────────────────────────────────────────
def compute_statistics(company_data: list, industry_data: list, all_records: list) -> dict:
    """计算汇总统计。"""
    # 公司数量：有 company 字段的顶层对象数
    company_count = len(company_data)

    # 行业数量：有 industry 字段的顶层对象数
    industry_count = len(industry_data)

    # 公司事件数量
    company_schema_count = sum(1 for r in all_records if r["target_type"] == "company")

    # 行业事件数量
    industry_schema_count = sum(1 for r in all_records if r["target_type"] == "industry")

    # 总记录数
    total_schema_count = len(all_records)

    # 唯一事件名称数
    unique_event_names = set(r["event_name"] for r in all_records)
    unique_event_name_count = len(unique_event_names)

    return {
        "company_count": company_count,
        "industry_count": industry_count,
        "company_schema_count": company_schema_count,
        "industry_schema_count": industry_schema_count,
        "total_schema_count": total_schema_count,
        "unique_event_name_count": unique_event_name_count,
    }


# ── 输出 ──────────────────────────────────────────────────
def write_jsonl(records: list, filepath: str):
    """写入 JSONL 文件。"""
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_excel(records: list, filepath: str):
    """写入 Excel 文件，列顺序固定。"""
    import pandas as pd

    columns = ["schema_id", "target_type", "target_code", "target_name", "event_name", "raw_description"]
    df = pd.DataFrame(records, columns=columns)
    df.to_excel(filepath, index=False, engine="openpyxl")


def write_summary(stats: dict, duplicates: list, empty_field_records: list, filepath: str):
    """写入汇总 JSON 报告。"""
    report = {
        **stats,
        "duplicate_schema_ids": duplicates,
        "empty_field_records": empty_field_records,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def write_frequency(records: list, filepath: str):
    """写入事件名称频次 CSV。"""
    event_names = [r["event_name"] for r in records]
    counter = Counter(event_names)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("event_name,count\n")
        for name, count in counter.most_common():
            # CSV 中如有逗号，用引号包裹
            escaped_name = name.replace('"', '""')
            f.write(f'"{escaped_name}",{count}\n')


# ── 主入口 ────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("事件 Schema 构建工具")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载原始数据...")
    company_data = load_json(COMPANY_FILE)
    industry_data = load_json(INDUSTRY_FILE)
    print(f"  公司记录数（顶层）: {len(company_data)}")
    print(f"  行业记录数（顶层）: {len(industry_data)}")

    # 2. 处理为公司/行业统一格式
    print("\n[2/6] 转换为统一 Schema 格式...")
    company_records = process_company_events(company_data)
    industry_records = process_industry_events(industry_data)
    all_records = company_records + industry_records
    print(f"  公司事件记录数: {len(company_records)}")
    print(f"  行业事件记录数: {len(industry_records)}")
    print(f"  总记录数: {len(all_records)}")

    # 3. 数据校验
    print("\n[3/6] 执行数据校验...")

    # 3a. 检查必需字段
    company_errors = validate_records(company_records, "company")
    industry_errors = validate_records(industry_records, "industry")
    all_field_errors = company_errors + industry_errors

    if all_field_errors:
        print(f"  ⚠ 发现 {len(all_field_errors)} 个字段异常：")
        for err in all_field_errors:
            print(f"    {err}")
    else:
        print("  ✓ 所有必需字段检查通过")

    # 3b. 检查重复 schema_id
    duplicates = check_duplicate_schema_ids(all_records)
    if duplicates:
        print(f"\n  ✗ 发现 {len(duplicates)} 个重复 schema_id：")
        for dup in duplicates:
            print(f"    {dup}")
        print("\n  ❌ 存在重复 schema_id，程序终止。")
        sys.exit(1)
    else:
        print("  ✓ schema_id 唯一性检查通过")

    # 3c. 收集空字段记录（用于报告）
    empty_field_records = []
    for rec in all_records:
        for field in ["schema_id", "target_type", "target_code", "target_name", "event_name", "raw_description"]:
            if rec.get(field) is None or (isinstance(rec.get(field), str) and rec[field].strip() == ""):
                empty_field_records.append({
                    "schema_id": rec.get("schema_id", "?"),
                    "field": field,
                })
    if empty_field_records:
        print(f"  ⚠ {len(empty_field_records)} 条记录存在空字段（详见报告）")
    else:
        print("  ✓ 所有字段非空检查通过")

    # 3d. 检查代码类型
    code_errors = []
    for rec in all_records:
        code = rec["target_code"]
        if isinstance(code, float):
            code_errors.append(f"{rec['schema_id']}: target_code 是 float ({code})")
        elif isinstance(code, str) and re.search(r'\.\d+$', code):
            code_errors.append(f"{rec['schema_id']}: target_code 疑似数值字符串 ({code})")
    if code_errors:
        print(f"  ⚠ {len(code_errors)} 条记录代码类型异常：")
        for ce in code_errors:
            print(f"    {ce}")
    else:
        print("  ✓ 所有代码格式检查通过")

    # 4. 计算统计
    print("\n[4/6] 生成统计...")
    stats = compute_statistics(company_data, industry_data, all_records)
    print(f"  公司数量: {stats['company_count']}")
    print(f"  行业数量: {stats['industry_count']}")
    print(f"  公司事件 Schema 数: {stats['company_schema_count']}")
    print(f"  行业事件 Schema 数: {stats['industry_schema_count']}")
    print(f"  Schema 总数: {stats['total_schema_count']}")
    print(f"  唯一事件名称数: {stats['unique_event_name_count']}")

    # 5. 输出文件
    print("\n[5/6] 写入输出文件...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    write_jsonl(all_records, OUTPUT_JSONL)
    print(f"  ✓ JSONL: {OUTPUT_JSONL}")

    write_excel(all_records, OUTPUT_XLSX)
    print(f"  ✓ Excel: {OUTPUT_XLSX}")

    write_summary(stats, duplicates, empty_field_records, SUMMARY_JSON)
    print(f"  ✓ 汇总报告: {SUMMARY_JSON}")

    write_frequency(all_records, FREQ_CSV)
    print(f"  ✓ 频次表: {FREQ_CSV}")

    # 6. 完成
    print("\n[6/6] 完成!")
    print("=" * 60)

    # 展示前 3 条
    print("\n前 3 条记录预览：")
    for i, rec in enumerate(all_records[:3]):
        print(f"\n--- 记录 {i+1} ---")
        for k, v in rec.items():
            display_v = v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v
            print(f"  {k}: {display_v}")


if __name__ == "__main__":
    main()
