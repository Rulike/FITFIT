#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Task1 Step 5.8-HC hierarchical taxonomy and audit package."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


ROOT = Path(__file__).resolve().parents[1]
BACKUP_ROOT = ROOT.parent / "backups"
OUT_DIR = ROOT / "outputs/task1_step5_8hc_hierarchical_event"
REPORT_JSON = ROOT / "reports/task1_step5_8hc_hierarchical_event_report.json"
REPORT_MD = ROOT / "reports/task1_step5_8hc_hierarchical_event_report.md"
DETAILS_XLSX = ROOT / "reports/task1_step5_8hc_hierarchical_event_details.xlsx"
COMPARE_XLSX = ROOT / "reports/task1_step5_8hc_flat_vs_hierarchical_comparison.xlsx"
HUMAN_REVIEW_XLSX = ROOT / "reports/task1_step5_8hc_taxonomy_human_review.xlsx"
TAXONOMY_MATRIX_XLSX = ROOT / "reports/task1_step5_8hc_taxonomy_matrix.xlsx"
ERRORS_XLSX = ROOT / "reports/task1_step5_8hc_hierarchical_errors.xlsx"
CONFIG_JSON = ROOT / "configs/task1_event_taxonomy_hierarchical_v1.json"
DOC_TAXONOMY = ROOT / "docs/TASK1_HIERARCHICAL_EVENT_TAXONOMY.md"
DOC_PIPELINE = ROOT / "docs/TASK1_HIERARCHICAL_MODEL_PIPELINE.md"
DOC_REPRO = ROOT / "docs/TASK1_HIERARCHICAL_REPRODUCTION_GUIDE.md"

OFFICIAL_TRAIN_XLSX = ROOT / "data/raw/news_with_tags_train.xlsx"
OFFICIAL_SCHEMA_JSONL = ROOT / "data/processed/event_schema.jsonl"
INCREMENTAL_XLSX = ROOT / "reports/财经新闻标注数据8.18.xlsx"
MANUAL_5000_PRIMARY_XLSX = ROOT / "data/raw/task1_hierarchical_manual_5000.xlsx"
MANUAL_5000_FALLBACK_XLSX = ROOT / "data/raw/2b421375-d014-442f-bb69-ff1ec00f7702.xlsx"
MANUAL_5000_LOCAL_XLSX = ROOT / "data/raw/标注5000条.xlsx"
TARGET_STOCK_JSON = ROOT / "data/raw/事件类型_标的.json"
HB_REPORT_JSON = ROOT / "outputs/task1_step5_8hb_event_mapping/mapping_manifest.json"
HB_LABEL_REPORT = ROOT / "outputs/task1_step5_8hb_event_mapping/label_level_mapping.json"
FROZEN_OFFICIAL_MODEL = ROOT / "outputs/task1_step5_8e_final_optimization/frozen_best_candidate/event_name_model.pkl"
FROZEN_EMOTION_MODEL = ROOT / "outputs/task1_step5_8f_final_candidate/frozen_emotion_duplication/event_emotion_model.pkl"
SHADOW_BASE_JSONL = ROOT / "outputs/task1_step5_8f_final_candidate/task1_official_delivery_candidate_v4.jsonl"


MERGED_SOURCE_TO_OFFICIAL = {
    "股票回购": ("资本市场运作", 0.96, "share repurchase is a direct capital-market operation in the official schema"),
    "分红送配": ("资本市场运作", 0.94, "cash dividend / stock dividend / rights allocation belongs to capital-market operation"),
    "定增募资": ("资本市场运作", 0.96, "private placement fund-raising is already covered by the official capital-market schema"),
    "并购重组": ("资本市场运作", 0.96, "merger / acquisition / restructuring is covered by the official capital-market schema"),
    "股权激励": ("资本市场运作", 0.96, "equity incentive is directly covered by the official capital-market schema"),
    "配售股份": ("资本市场运作", 0.95, "share placement is a capital-market financing action"),
    "IPO上市": ("资本市场运作", 0.90, "IPO is a capital-market financing / listing action"),
    "合同签订": ("签订重要合同", 0.97, "contract signing is the same event core as the official contract schema"),
    "产品发布": ("新产品升级发布", 0.95, "product launch is the same event core as the official product-release schema"),
    "技术突破": ("技术突破与创新", 0.96, "technical breakthrough is the same event core as the official innovation schema"),
    "产能扩张": ("产能布局调整", 0.94, "capacity expansion is the same capacity-layout event core"),
    "价格波动": ("大宗商品景气与价格波动加剧", 0.90, "commodity price fluctuation is the same industry price event core"),
    "供需变化": ("行业下游的需求变化", 0.90, "supply-demand change is the same downstream demand event core"),
}

EXTENDED_SOURCE_LABELS = {
    "中标项目",
    "战略合作",
    "产品认证",
    "业绩增长",
    "业绩下降",
    "业绩亏损",
    "销量增长",
    "大股东增持",
    "大股东减持",
    "股份质押",
    "股东大会",
    "高管变动",
    "股权变动",
    "监管处罚",
    "诉讼纠纷",
    "安全事故",
    "退市风险",
    "退市风险警示",
    "政策利好",
    "政策监管",
    "国际市场变化",
    "大股东增减持",
    "宏观经济数据",
    "技术趋势",
    "行业整合",
    "行业景气变化",
}

MARKET_SIGNAL_LABELS = {
    "股价上涨",
    "股价下跌",
    "股价异动",
    "涨停板",
    "跌停板",
    "创历史新高",
    "创历史新低",
    "资金流入",
    "资金流出",
    "龙虎榜",
    "融资融券",
}

UNRESOLVED_LABELS = {"行业数据", "行业数据发布"}

LABEL_KIND_PRIORITY = {
    **{k: "official_merge" for k in MERGED_SOURCE_TO_OFFICIAL},
    **{k: "extended_leaf" for k in EXTENDED_SOURCE_LABELS},
    **{k: "market_signal" for k in MARKET_SIGNAL_LABELS},
    **{k: "unresolved_meta" for k in UNRESOLVED_LABELS},
}

SOURCE_LABEL_DEFINITION = {
    "中标项目": "企业通过竞标获得项目或订单，通常对应明确的中标公告、项目确认或合同签署前置动作。",
    "战略合作": "企业与合作方建立战略合作、联合开发或渠道协同关系，强调长期业务协同而非单纯合同签署。",
    "产品认证": "产品通过资质认证、标准认证、审评审批或测试验证，体现合规准入与商业化门槛突破。",
    "业绩增长": "公司经营结果改善，营收、利润、毛利率或销量出现增长。",
    "业绩下降": "公司经营结果走弱，营收、利润、毛利率或销量出现下滑。",
    "业绩亏损": "公司报告期出现亏损、亏损扩大或由盈转亏。",
    "销量增长": "产品销量、出货量、交付量或销售规模上升。",
    "大股东增持": "控股股东、实控人或重要股东通过二级市场或协议方式增加持股。",
    "大股东减持": "控股股东、实控人或重要股东通过二级市场或协议方式减少持股。",
    "股份质押": "股东将股份用于质押融资、补充担保或流动性安排。",
    "股东大会": "公司召开股东大会并形成表决、授权或审议结果。",
    "高管变动": "董事、监事、高管或核心管理团队发生任免、辞职或调整。",
    "股权变动": "公司控制权、股权结构或持股比例发生变化。",
    "监管处罚": "监管机关对公司或相关主体作出处罚、处分、警示或问责。",
    "诉讼纠纷": "公司因合同、侵权、劳动、债务或其他事项进入诉讼、仲裁或纠纷处理阶段。",
    "安全事故": "生产、运营、运输、施工或实验环节发生安全事故或重大异常事件。",
    "退市风险": "公司因财务、合规、交易或持续经营问题面临退市风险。",
    "退市风险警示": "公司被实施退市风险警示、ST/*ST 或类似风险提示。",
    "政策利好": "政策、规划、补贴或监管调整对行业或企业形成偏正向支持。",
    "政策监管": "政策、规范或监管措施对行业或企业形成约束、规范或收紧。",
    "国际市场变化": "海外市场、跨境贸易、国际需求或全球竞争环境发生变化。",
    "大股东增减持": "大股东、控股股东或重要股东出现增持或减持动作，标签未单独拆分方向。",
    "宏观经济数据": "GDP、通胀、利率、PMI、进出口等宏观统计数据或相关解读发布。",
    "技术趋势": "围绕某一技术路线、工具链、方法论或行业技术范式的趋势判断、路线变化或行业观察。",
    "行业整合": "行业集中度提升、兼并重组、资源整合或竞争格局再平衡。",
    "行业景气变化": "行业景气度、供需周期、景气回暖或走弱的变化。",
    "行业数据发布": "行业级数据发布、监测指标、统计口径或观察口径说明，缺乏稳定的事件核心。",
}

EXTENDED_OFFICIAL_CANDIDATE = {
    "中标项目": ("签订重要合同", 0.86, "winning a bid usually precedes a concrete contract / project delivery event"),
    "战略合作": ("重大业务拓展", 0.80, "strategic cooperation often corresponds to a broader business expansion event"),
    "产品认证": ("产品技术创新", 0.78, "certification often accompanies product innovation or technical validation"),
    "政策利好": ("行业政策推动", 0.74, "policy support is close to sector policy promotion in the official schema"),
    "政策监管": ("行业调控政策发布", 0.74, "regulatory policy is close to sector control policy in the official schema"),
}

L1_ORDER = [
    "股权与资本运作",
    "公司治理与股东行为",
    "合同、交易与重大项目",
    "产品、技术与研发",
    "产能、供应链与生产经营",
    "企业战略与市场扩张",
    "经营业绩与财务风险",
    "法律、监管与安全风险",
    "行业供需与市场价格",
    "政策与宏观环境",
    "地缘、公共与外部冲击",
    "市场交易信号",
    "待确认元标签",
]

L1_SLUG = {
    "股权与资本运作": "equity_capital_ops",
    "公司治理与股东行为": "governance_shareholder_behaviour",
    "合同、交易与重大项目": "contract_trade_projects",
    "产品、技术与研发": "product_tech_rnd",
    "产能、供应链与生产经营": "capacity_supply_chain_ops",
    "企业战略与市场扩张": "strategy_market_expansion",
    "经营业绩与财务风险": "performance_financial_risk",
    "法律、监管与安全风险": "legal_regulatory_safety_risk",
    "行业供需与市场价格": "industry_supply_market_price",
    "政策与宏观环境": "policy_macro_environment",
    "地缘、公共与外部冲击": "geopolitical_public_external_shock",
    "市场交易信号": "market_signal",
    "待确认元标签": "unresolved_meta",
}

L2_GROUPS = [
    ("资本工具运作", "capital_tools", "股权与资本运作"),
    ("股东与治理行动", "shareholder_governance", "公司治理与股东行为"),
    ("合同签订", "contract_signing", "合同、交易与重大项目"),
    ("中标与合作", "bid_cooperation", "合同、交易与重大项目"),
    ("项目投资与建设", "project_investment", "合同、交易与重大项目"),
    ("产品发布与迭代", "product_launch", "产品、技术与研发"),
    ("认证与准入", "certification_entry", "产品、技术与研发"),
    ("研发与技术突破", "rnd_tech_breakthrough", "产品、技术与研发"),
    ("产能布局与供应链", "capacity_supply_chain", "产能、供应链与生产经营"),
    ("经营策略调整", "operation_strategy_adjustment", "产能、供应链与生产经营"),
    ("战略扩张与出海", "strategy_expansion", "企业战略与市场扩张"),
    ("业务转型与新赛道", "business_transformation", "企业战略与市场扩张"),
    ("业绩与利润变化", "performance_change", "经营业绩与财务风险"),
    ("现金流与偿债风险", "cashflow_risk", "经营业绩与财务风险"),
    ("合规与处罚风险", "compliance_penalty", "法律、监管与安全风险"),
    ("诉讼与安全风险", "litigation_safety", "法律、监管与安全风险"),
    ("退市与风险警示", "delisting_warning", "法律、监管与安全风险"),
    ("商品价格与景气", "commodity_price", "行业供需与市场价格"),
    ("需求变化与竞争", "demand_competition", "行业供需与市场价格"),
    ("产业政策", "industry_policy", "政策与宏观环境"),
    ("宏观政策", "macro_policy", "政策与宏观环境"),
    ("地缘与公共冲击", "geo_public_shock", "地缘、公共与外部冲击"),
    ("价格信号", "price_signal", "市场交易信号"),
    ("资金与榜单", "capital_flow_rank", "市场交易信号"),
    ("未决元标签", "pending_meta", "待确认元标签"),
]

L2_SLUG = {name: slug for name, slug, _ in L2_GROUPS}
L2_PARENT = {name: parent for name, _, parent in L2_GROUPS}


OFFICIAL_LABEL_OVERRIDES = {
    "资本市场运作": ("股权与资本运作", "资本工具运作"),
    "现金流风险": ("经营业绩与财务风险", "现金流与偿债风险"),
    "产品技术创新": ("产品、技术与研发", "研发与技术突破"),
    "产能布局调整": ("产能、供应链与生产经营", "产能布局与供应链"),
    "新产品升级发布": ("产品、技术与研发", "产品发布与迭代"),
    "技术突破与创新": ("产品、技术与研发", "研发与技术突破"),
    "技术研发进展": ("产品、技术与研发", "研发与技术突破"),
    "关键技术突破": ("产品、技术与研发", "研发与技术突破"),
    "医药研发突破": ("产品、技术与研发", "研发与技术突破"),
    "动力电池产品迭代": ("产品、技术与研发", "产品发布与迭代"),
    "储能电池产品迭代": ("产品、技术与研发", "产品发布与迭代"),
    "光伏新技术迭代升级": ("产品、技术与研发", "研发与技术突破"),
    "光伏组件行业技术迭代": ("产品、技术与研发", "研发与技术突破"),
    "钢铁生产核心技术突破": ("产品、技术与研发", "研发与技术突破"),
    "化学制品科技创新": ("产品、技术与研发", "研发与技术突破"),
    "业务拓展": ("企业战略与市场扩张", "战略扩张与出海"),
    "新业务拓展": ("企业战略与市场扩张", "业务转型与新赛道"),
    "重大业务拓展": ("企业战略与市场扩张", "战略扩张与出海"),
    "公司战略调整": ("企业战略与市场扩张", "业务转型与新赛道"),
    "公司战略变化及对外投资": ("企业战略与市场扩张", "战略扩张与出海"),
    "业务战略调整": ("企业战略与市场扩张", "业务转型与新赛道"),
    "企业转型催化": ("企业战略与市场扩张", "业务转型与新赛道"),
    "全球化市场扩张与品牌布局": ("企业战略与市场扩张", "战略扩张与出海"),
    "国际化布局": ("企业战略与市场扩张", "战略扩张与出海"),
    "To B业务拓展（新能源、机器人等）": ("企业战略与市场扩张", "业务转型与新赛道"),
    "数字化转型深化": ("企业战略与市场扩张", "业务转型与新赛道"),
    "结构升级": ("产品、技术与研发", "产品发布与迭代"),
    "调改措施进展": ("产能、供应链与生产经营", "经营策略调整"),
    "零售业务策略调整": ("产能、供应链与生产经营", "经营策略调整"),
    "合资品牌战略": ("企业战略与市场扩张", "战略扩张与出海"),
    "商业航天与低空经济发展": ("合同、交易与重大项目", "项目投资与建设"),
    "新车出海": ("企业战略与市场扩张", "战略扩张与出海"),
    "新车发布定价": ("产品、技术与研发", "产品发布与迭代"),
    "海外产能进展": ("产能、供应链与生产经营", "产能布局与供应链"),
    "重大项目投资建设": ("合同、交易与重大项目", "项目投资与建设"),
    "重大基建项目开发": ("合同、交易与重大项目", "项目投资与建设"),
    "签订重要合同": ("合同、交易与重大项目", "合同签订"),
    "中标项目": ("合同、交易与重大项目", "中标与合作"),
    "战略合作": ("合同、交易与重大项目", "中标与合作"),
    "业绩增长": ("经营业绩与财务风险", "业绩与利润变化"),
    "业绩下降": ("经营业绩与财务风险", "业绩与利润变化"),
    "业绩亏损": ("经营业绩与财务风险", "业绩与利润变化"),
    "销量增长": ("经营业绩与财务风险", "业绩与利润变化"),
    "大股东增持": ("公司治理与股东行为", "股东与治理行动"),
    "大股东减持": ("公司治理与股东行为", "股东与治理行动"),
    "股份质押": ("公司治理与股东行为", "股东与治理行动"),
    "股东大会": ("公司治理与股东行为", "股东与治理行动"),
    "高管变动": ("公司治理与股东行为", "股东与治理行动"),
    "股权变动": ("公司治理与股东行为", "股东与治理行动"),
    "监管处罚": ("法律、监管与安全风险", "合规与处罚风险"),
    "诉讼纠纷": ("法律、监管与安全风险", "诉讼与安全风险"),
    "安全事故": ("法律、监管与安全风险", "诉讼与安全风险"),
    "退市风险": ("法律、监管与安全风险", "退市与风险警示"),
    "退市风险警示": ("法律、监管与安全风险", "退市与风险警示"),
    "政策利好": ("政策与宏观环境", "产业政策"),
    "政策监管": ("政策与宏观环境", "产业政策"),
    "行业政策": ("政策与宏观环境", "产业政策"),
    "行业政策推动": ("政策与宏观环境", "产业政策"),
    "行业调控政策发布": ("政策与宏观环境", "产业政策"),
    "稳增长政策": ("政策与宏观环境", "宏观政策"),
    "金融政策支持": ("政策与宏观环境", "宏观政策"),
    "绿色政策发布": ("政策与宏观环境", "产业政策"),
    "电力投资政策": ("政策与宏观环境", "产业政策"),
    "农业政策变动": ("政策与宏观环境", "产业政策"),
    "影视行业政策变动": ("政策与宏观环境", "产业政策"),
    "教育行业政策调整": ("政策与宏观环境", "产业政策"),
    "国内创新药支持政策": ("政策与宏观环境", "产业政策"),
    "工业自动化政策支持变化": ("政策与宏观环境", "产业政策"),
    "化债政策": ("政策与宏观环境", "宏观政策"),
    "存款利率下调": ("政策与宏观环境", "宏观政策"),
    "发生地缘政治事件": ("地缘、公共与外部冲击", "地缘与公共冲击"),
    "地缘政治": ("地缘、公共与外部冲击", "地缘与公共冲击"),
    "公共卫生事件": ("地缘、公共与外部冲击", "地缘与公共冲击"),
    "OPEC+产量政策变化": ("行业供需与市场价格", "商品价格与景气"),
    "大宗商品景气与价格波动加剧": ("行业供需与市场价格", "商品价格与景气"),
    "水产品价格变动": ("行业供需与市场价格", "商品价格与景气"),
    "行业下游的需求变化": ("行业供需与市场价格", "需求变化与竞争"),
    "市场竞争": ("行业供需与市场价格", "需求变化与竞争"),
    "市场竞争加剧": ("行业供需与市场价格", "需求变化与竞争"),
    "影视市场表现变化": ("行业供需与市场价格", "需求变化与竞争"),
    "航海装备产业升级": ("企业战略与市场扩张", "业务转型与新赛道"),
}

MARKET_SIGNAL_L2 = {
    "股价上涨": "价格信号",
    "股价下跌": "价格信号",
    "股价异动": "价格信号",
    "涨停板": "价格信号",
    "跌停板": "价格信号",
    "创历史新高": "价格信号",
    "创历史新低": "价格信号",
    "资金流入": "资金与榜单",
    "资金流出": "资金与榜单",
    "龙虎榜": "资金与榜单",
    "融资融券": "资金与榜单",
}

SOURCE_LABEL_ORDER = [
    "合同签订",
    "中标项目",
    "战略合作",
    "产品发布",
    "产品认证",
    "技术突破",
    "产能扩张",
    "价格波动",
    "供需变化",
    "政策利好",
    "政策监管",
    "业绩增长",
    "业绩下降",
    "业绩亏损",
    "销量增长",
    "大股东增持",
    "大股东减持",
    "股份质押",
    "股东大会",
    "高管变动",
    "股权变动",
    "监管处罚",
    "诉讼纠纷",
    "安全事故",
    "退市风险",
    "退市风险警示",
    "股票回购",
    "分红送配",
    "IPO上市",
    "并购重组",
    "定增募资",
    "股权激励",
    "配售股份",
    "股价上涨",
    "股价下跌",
    "股价异动",
    "涨停板",
    "跌停板",
    "创历史新高",
    "创历史新低",
    "资金流入",
    "资金流出",
    "龙虎榜",
    "融资融券",
    "行业数据",
]


def discover_source_label_order(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for label in SOURCE_LABEL_ORDER:
        if label not in seen:
            order.append(label)
            seen.add(label)
    for row in rows:
        label = norm_text(row.get("source_event_name") or row.get("event_name"))
        if label and label not in seen:
            order.append(label)
            seen.add(label)
    return order


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def cell_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def norm_text(value: Any) -> str:
    value = cell_value(value)
    if value == "":
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def compact_text(value: Any, limit: int = 240) -> str:
    return norm_text(value)[:limit]


def text_sha(text: Any) -> str:
    return sha256_text(norm_text(text))


def source_text(row: dict[str, Any]) -> str:
    parts = [
        norm_text(row.get("article_title")),
        norm_text(row.get("article_source")),
        norm_text(row.get("target_name")),
        norm_text(row.get("content")),
    ]
    return "\n".join(p for p in parts if p)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def label_kind(label: str) -> str:
    return LABEL_KIND_PRIORITY.get(label, "extended_leaf")


def l1_l2_for_label(label: str) -> tuple[str, str]:
    if label in MARKET_SIGNAL_LABELS:
        return "市场交易信号", MARKET_SIGNAL_L2[label]
    if label in UNRESOLVED_LABELS:
        return "待确认元标签", "未决元标签"
    if label in MERGED_SOURCE_TO_OFFICIAL:
        target = MERGED_SOURCE_TO_OFFICIAL[label][0]
        return OFFICIAL_LABEL_OVERRIDES.get(target, _infer_l1_l2_by_name(target))
    if label in EXTENDED_SOURCE_LABELS:
        if label in {"大股东增持", "大股东减持", "股份质押", "股东大会", "高管变动", "股权变动"}:
            return "公司治理与股东行为", "股东与治理行动"
        if label in {"业绩增长", "业绩下降", "业绩亏损", "销量增长"}:
            return "经营业绩与财务风险", "业绩与利润变化"
        if label in {"监管处罚"}:
            return "法律、监管与安全风险", "合规与处罚风险"
        if label in {"诉讼纠纷", "安全事故", "退市风险", "退市风险警示"}:
            return "法律、监管与安全风险", "诉讼与安全风险" if label != "退市风险警示" and label != "退市风险" else "退市与风险警示"
        if label in {"中标项目", "战略合作"}:
            return "合同、交易与重大项目", "中标与合作"
        if label in {"产品认证"}:
            return "产品、技术与研发", "认证与准入"
        if label in {"政策利好", "政策监管"}:
            return "政策与宏观环境", "产业政策"
    return _infer_l1_l2_by_name(label)


def _infer_l1_l2_by_name(label: str) -> tuple[str, str]:
    if label == "国际市场变化":
        return "地缘、公共与外部冲击", "地缘与公共冲击"
    if label == "宏观经济数据":
        return "政策与宏观环境", "宏观政策"
    if label == "技术趋势":
        return "产品、技术与研发", "研发与技术突破"
    if label == "行业整合":
        return "行业供需与市场价格", "需求变化与竞争"
    if label == "行业景气变化":
        return "行业供需与市场价格", "商品价格与景气"
    if any(k in label for k in ["股价", "资金流", "龙虎榜", "融资融券", "涨停", "跌停"]):
        return "市场交易信号", MARKET_SIGNAL_L2.get(label, "价格信号")
    if any(k in label for k in ["回购", "分红", "定增", "并购", "配售", "IPO", "股权激励", "资本市场运作"]):
        return "股权与资本运作", "资本工具运作"
    if any(k in label for k in ["增持", "减持", "质押", "股东大会", "高管", "股权变动"]):
        return "公司治理与股东行为", "股东与治理行动"
    if any(k in label for k in ["合同", "中标", "合作", "项目", "基建", "商业航天", "低空"]):
        return "合同、交易与重大项目", "中标与合作" if any(k in label for k in ["中标", "合作"]) else "项目投资与建设"
    if any(k in label for k in ["产品", "技术", "研发", "创新", "认证", "电池", "光伏", "钢铁", "化学", "医药"]):
        return "产品、技术与研发", "研发与技术突破" if any(k in label for k in ["技术", "研发", "创新"]) else "产品发布与迭代"
    if any(k in label for k in ["产能", "供应链", "出海", "零售", "调改", "合资品牌", "结构升级", "新车"]):
        return "产能、供应链与生产经营", "经营策略调整"
    if any(k in label for k in ["战略", "业务拓展", "新业务", "转型", "国际化", "全球化", "To B", "航海装备", "商业航天", "低空经济"]):
        return "企业战略与市场扩张", "战略扩张与出海" if any(k in label for k in ["国际化", "全球化", "出海", "商业航天", "低空"]) else "业务转型与新赛道"
    if any(k in label for k in ["业绩", "现金流", "亏损", "销量"]):
        return "经营业绩与财务风险", "业绩与利润变化" if "现金流" not in label else "现金流与偿债风险"
    if any(k in label for k in ["监管", "处罚", "诉讼", "安全", "退市"]):
        return "法律、监管与安全风险", "退市与风险警示" if "退市" in label else "合规与处罚风险" if "监管" in label or "处罚" in label else "诉讼与安全风险"
    if any(k in label for k in ["政策", "利率", "化债", "稳增长"]):
        return "政策与宏观环境", "宏观政策" if any(k in label for k in ["利率", "化债", "稳增长"]) else "产业政策"
    if any(k in label for k in ["地缘", "公共卫生"]):
        return "地缘、公共与外部冲击", "地缘与公共冲击"
    if any(k in label for k in ["价格", "供需", "需求", "竞争", "景气", "市场表现"]):
        return "行业供需与市场价格", "商品价格与景气" if "价格" in label or "景气" in label else "需求变化与竞争"
    return "企业战略与市场扩张", "业务转型与新赛道"


def source_label_target_kind(label: str, official_names: set[str] | None = None) -> str:
    if official_names and label in official_names:
        return "official_merge"
    if label in MARKET_SIGNAL_LABELS:
        return "market_signal"
    if label in UNRESOLVED_LABELS:
        return "unresolved_meta"
    if label in EXTENDED_SOURCE_LABELS:
        return "extended_leaf"
    if label in MERGED_SOURCE_TO_OFFICIAL:
        return "official_merge"
    return "extended_leaf"


def official_split_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    dsu = DSU()
    for r in rows:
        article_node = f"article:{r['row_id']}"
        row_node = f"row:{r['row_id']}"
        dsu.union(article_node, row_node)
        dup = str(r.get("duplication_id") or "")
        if dup:
            dsu.union(article_node, f"dup:{dup}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[dsu.find(f"row:{r['row_id']}")].append(r)

    rng = np.random.RandomState(2026)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda kv: (-len(kv[1]), min(x["article_file_id"] for x in kv[1])))

    targets = {"train": 0.70 * len(rows), "model_selection_dev": 0.15 * len(rows), "untouched_internal_holdout": 0.15 * len(rows)}
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "model_selection_dev": [], "untouched_internal_holdout": []}
    event_counts_total = Counter(r["event_name"] for r in rows)

    rare_group_ids = set()
    for gid, rs in group_items:
        labels = {r["event_name"] for r in rs}
        if any(event_counts_total[l] < 3 for l in labels):
            rare_group_ids.add(gid)

    for gid, rs in group_items:
        if gid in rare_group_ids:
            split_rows["train"].extend(rs)
            continue
        deficits = {s: targets[s] - len(split_rows[s]) for s in split_rows}
        chosen = max(deficits.items(), key=lambda kv: (kv[1], -len(split_rows[kv[0]]), kv[0]))[0]
        if deficits[chosen] <= 0:
            chosen = min(split_rows, key=lambda s: len(split_rows[s]) / max(1, targets[s]))
        split_rows[chosen].extend(rs)

    split_ids = {s: [r["row_id"] for r in rs] for s, rs in split_rows.items()}
    manifest = {
        "seed": 2026,
        "method": "article_duplication_connected_components_greedy_size_balance",
        "counts": {s: len(v) for s, v in split_rows.items()},
        "event_class_counts": {s: dict(Counter(r["event_name"] for r in v)) for s, v in split_rows.items()},
        "rare_classes_forced_train": sorted([k for k, v in event_counts_total.items() if v < 3]),
    }
    return split_ids, manifest


class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_official_schema() -> dict[str, dict[str, Any]]:
    rows = read_jsonl(OFFICIAL_SCHEMA_JSONL)
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_name[norm_text(row["event_name"])].append(row)
    catalog: dict[str, dict[str, Any]] = {}
    for name, items in sorted(by_name.items()):
        descriptions = [norm_text(i.get("raw_description")) for i in items if norm_text(i.get("raw_description"))]
        catalog[name] = {
            "event_name": name,
            "target_types": sorted({norm_text(i.get("target_type")) for i in items}),
            "schema_ids": [norm_text(i.get("schema_id")) for i in items],
            "descriptions": descriptions[:8],
            "description_excerpt": descriptions[0][:1000] if descriptions else "",
            "official_schema_rows": items,
        }
    if len(catalog) != 64:
        raise RuntimeError(f"official schema must contain 64 unique event names, got {len(catalog)}")
    return catalog


def load_official_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sheet in ["个股新闻", "行业新闻"]:
        df = pd.read_excel(OFFICIAL_TRAIN_XLSX, sheet_name=sheet).fillna("")
        for idx, r in df.iterrows():
            row = {k: cell_value(v) for k, v in r.to_dict().items()}
            row["sheet"] = sheet
            row["source_dataset"] = "official2081"
            row["target_type"] = "company" if sheet == "个股新闻" else "industry"
            row["row_id"] = f"{sheet}_{idx:05d}"
            row["article_file_id"] = str(row["article_file_id"])
            row["event_name"] = str(row["event_name"])
            row["event_emotion"] = str(row["event_emotion"])
            row["duplication_id"] = str(row.get("duplication_id") or "")
            row["content_sha256"] = text_sha(row.get("content"))
            row["title_sha256"] = text_sha(row.get("article_title"))
            row["target_key"] = f"{row['target_type']}:{row.get('trading_code', row.get('industry_code', ''))}:{row.get('secu_abbr', row.get('industry_name', ''))}"
            rows.append(row)
    return rows


def load_incremental_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    xls = pd.ExcelFile(INCREMENTAL_XLSX)
    expected = {"个股新闻", "行业新闻"}
    if set(xls.sheet_names) != expected:
        raise RuntimeError(f"unexpected incremental sheets: {xls.sheet_names}")
    rows: list[dict[str, Any]] = []
    sheet_stats = {}
    for sheet in ["个股新闻", "行业新闻"]:
        df = pd.read_excel(INCREMENTAL_XLSX, sheet_name=sheet, dtype=object).fillna("")
        target_type = "company" if sheet == "个股新闻" else "industry"
        sheet_stats[sheet] = {"rows": len(df), "columns": list(df.columns)}
        for idx, raw in df.iterrows():
            item = {col: cell_value(raw.get(col)) for col in df.columns}
            item.update(
                {
                    "row_number_1based": int(idx) + 2,
                    "original_sheet": sheet,
                    "source_dataset": "incremental1000",
                    "target_type": target_type,
                    "target_name": norm_text(raw.get("secu_abbr" if target_type == "company" else "industry_name")),
                    "source_event_name": norm_text(raw.get("event_name")),
                    "content_sha256": text_sha(raw.get("content")),
                    "title_sha256": text_sha(raw.get("article_title")),
                    "duplication_id_norm": norm_text(raw.get("duplication_id")),
                }
            )
            rows.append(item)
    if len(rows) != 1000:
        raise RuntimeError(f"expected 1000 rows, got {len(rows)}")
    ids = [norm_text(r.get("article_file_id")) for r in rows]
    if len(set(ids)) != 1000:
        raise RuntimeError("article_file_id must be unique across incremental workbook")
    stats = {
        "sha256": sha256_file(INCREMENTAL_XLSX),
        "rows": len(rows),
        "unique_article_file_id": len(set(ids)),
        "sheet_stats": sheet_stats,
        "source_event_name_count": len({r["source_event_name"] for r in rows}),
        "source_event_name_distribution": dict(Counter(r["source_event_name"] for r in rows)),
        "target_type_distribution": dict(Counter(r["target_type"] for r in rows)),
        "emotion_distribution": dict(Counter(norm_text(r.get("event_emotion")) for r in rows)),
    }
    return rows, stats


def load_hb_report() -> dict[str, Any]:
    return read_json(HB_REPORT_JSON) if HB_REPORT_JSON.exists() else {}


def load_hb_label_report() -> list[dict[str, Any]]:
    return read_json(HB_LABEL_REPORT) if HB_LABEL_REPORT.exists() else []


def resolve_manual_5000_xlsx() -> Path:
    for candidate in [MANUAL_5000_PRIMARY_XLSX, MANUAL_5000_FALLBACK_XLSX, MANUAL_5000_LOCAL_XLSX]:
        if candidate.exists():
            return candidate
    discovered = sorted(ROOT.rglob("2b421375-d014-442f-bb69-ff1ec00f7702.xlsx"))
    if discovered:
        return discovered[0]
    raise FileNotFoundError(
        "manual 5000 workbook not found; checked "
        f"{MANUAL_5000_PRIMARY_XLSX}, {MANUAL_5000_FALLBACK_XLSX}, {MANUAL_5000_LOCAL_XLSX}"
    )


def load_security_master_pairs() -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    if TARGET_STOCK_JSON.exists():
        try:
            data = json.loads(TARGET_STOCK_JSON.read_text(encoding="utf-8"))
            for item in data if isinstance(data, list) else []:
                company = item.get("company", {}) if isinstance(item, dict) else {}
                pairs.add((norm_text(company.get("trading_code")), norm_text(company.get("secu_abbr"))))
        except Exception:
            pass
    try:
        df = pd.read_excel(OFFICIAL_TRAIN_XLSX, sheet_name="个股新闻", dtype=object).fillna("")
        for _, row in df.iterrows():
            pairs.add((norm_text(row.get("trading_code")), norm_text(row.get("secu_abbr"))))
    except Exception:
        pass
    return {(code, name) for code, name in pairs if code and name}


def derive_safe_trading_code(
    raw_code: Any,
    title: Any,
    content: Any,
    secu_abbr: Any,
    master_pairs: set[tuple[str, str]],
) -> tuple[str, bool, str]:
    code = re.sub(r"\.0$", "", norm_text(raw_code))
    if not (re.fullmatch(r"\d{6}", code) or re.fullmatch(r"0\d{4}", code)):
        return "", False, "unsafe_format"
    text = f"{norm_text(title)} {norm_text(content)} {norm_text(secu_abbr)}"
    if code in text or (code.startswith("0") and code[1:] in text):
        return code, True, "content_proved"
    if (code, norm_text(secu_abbr)) in master_pairs:
        return code, True, "master_pair_proved"
    return "", False, "not_proven"


def load_manual_5000_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = resolve_manual_5000_xlsx()
    xls = pd.ExcelFile(path)
    expected = {"个股新闻", "行业新闻"}
    if set(xls.sheet_names) != expected:
        raise RuntimeError(f"unexpected manual 5000 sheets: {xls.sheet_names}")
    master_pairs = load_security_master_pairs()
    rows: list[dict[str, Any]] = []
    sheet_stats = {}
    required = {
        "个股新闻": [
            "article_file_id",
            "article_title",
            "article_publish_time",
            "content",
            "Unnamed: 5",
            "secu_abbr",
            "industry_code",
            "industry_name",
            "event_name",
            "event_emotion",
            "event_impact_analysis",
            "duplication_id",
        ],
        "行业新闻": [
            "article_file_id",
            "article_title",
            "article_publish_time",
            "content",
            "industry_code",
            "industry_name",
            "event_name",
            "event_emotion",
            "event_impact_analysis",
            "duplication_id",
        ],
    }
    for sheet in ["个股新闻", "行业新闻"]:
        df = pd.read_excel(path, sheet_name=sheet, dtype=object).fillna("")
        target_type = "company" if sheet == "个股新闻" else "industry"
        missing_required = {
            col: int((df[col].astype(str).str.strip() == "").sum()) if col in df.columns else len(df)
            for col in required[sheet]
        }
        if any(v > 0 for v in missing_required.values()):
            raise RuntimeError(f"manual 5000 workbook has missing required fields in {sheet}: {missing_required}")
        sheet_stats[sheet] = {"rows": len(df), "columns": list(df.columns), "missing_required": missing_required}
        for idx, raw in df.iterrows():
            item = {col: cell_value(raw.get(col)) for col in df.columns}
            item.update(
                {
                    "row_number_1based": int(idx) + 2,
                    "original_sheet": sheet,
                    "source_dataset": "manual5000",
                    "target_type": target_type,
                    "target_name": norm_text(raw.get("secu_abbr" if target_type == "company" else "industry_name")),
                    "source_event_name": norm_text(raw.get("event_name")),
                    "content_sha256": text_sha(raw.get("content")),
                    "title_sha256": text_sha(raw.get("article_title")),
                    "duplication_id_norm": norm_text(raw.get("duplication_id")),
                    "security_code_raw": norm_text(raw.get("Unnamed: 5")) if target_type == "company" else "",
                }
            )
            if target_type == "company":
                safe_code, proven, reason = derive_safe_trading_code(
                    raw.get("Unnamed: 5"),
                    raw.get("article_title"),
                    raw.get("content"),
                    raw.get("secu_abbr"),
                    master_pairs,
                )
                item["trading_code"] = safe_code
                item["trading_code_proven"] = proven
                item["trading_code_proof"] = reason
            else:
                item["trading_code"] = ""
                item["trading_code_proven"] = False
                item["trading_code_proof"] = ""
            rows.append(item)
    if len(rows) <= 0:
        raise RuntimeError("manual workbook contains no rows")
    ids = [norm_text(r.get("article_file_id")) for r in rows]
    if len(set(ids)) != len(rows):
        raise RuntimeError(
            f"article_file_id must be unique across manual workbook: "
            f"rows={len(rows)}, unique_ids={len(set(ids))}"
        )
    stats = {
        "workbook": rel(path) if path.is_relative_to(ROOT) else str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "unique_article_file_id": len(set(ids)),
        "sheet_stats": sheet_stats,
        "source_event_name_count": len({r["source_event_name"] for r in rows}),
        "source_event_name_distribution": dict(Counter(r["source_event_name"] for r in rows)),
        "target_type_distribution": dict(Counter(r["target_type"] for r in rows)),
        "emotion_distribution": dict(Counter(norm_text(r.get("event_emotion")) for r in rows)),
        "company_trading_code_proven_count": int(sum(1 for r in rows if r["target_type"] == "company" and r.get("trading_code_proven"))),
        "company_trading_code_safe_count": int(sum(1 for r in rows if r["target_type"] == "company" and r.get("trading_code"))),
        "company_trading_code_unsafe_count": int(sum(1 for r in rows if r["target_type"] == "company" and not r.get("trading_code"))),
        "internal_duplicate_title_sha_count": int(sum(v > 1 for v in Counter(r["title_sha256"] for r in rows).values())),
        "internal_duplicate_content_sha_count": int(sum(v > 1 for v in Counter(r["content_sha256"] for r in rows).values())),
        "internal_duplicate_duplication_id_count": int(sum(v > 1 for v in Counter(r["duplication_id_norm"] for r in rows if r["duplication_id_norm"]).values())),
    }
    return rows, stats


def load_shadow_input_rows() -> list[dict[str, Any]]:
    return read_jsonl(SHADOW_BASE_JSONL)


def resolve_source_spec(label: str, official_names: set[str] | None = None) -> dict[str, Any]:
    kind = source_label_target_kind(label, official_names)
    is_official_exact = bool(official_names and label in official_names)
    l1, l2 = l1_l2_for_label(label)
    target_scope = "both"
    if label in {"龙虎榜", "融资融券", "资金流入", "资金流出", "股价上涨", "股价下跌", "股价异动", "涨停板", "跌停板", "创历史新高", "创历史新低"}:
        target_scope = "company"
    elif label in UNRESOLVED_LABELS or label in {"国际市场变化", "宏观经济数据", "技术趋势", "行业整合", "行业景气变化"}:
        target_scope = "industry"
    elif label == "大股东增减持":
        target_scope = "company"
    official_name = ""
    confidence = 0.0
    merge_reason = ""
    candidate_official_name = ""
    candidate_confidence = 0.0
    candidate_reason = ""
    if is_official_exact:
        official_name = label
        confidence = 1.0
        merge_reason = "source label matches an official 64 label exactly"
    elif label in MERGED_SOURCE_TO_OFFICIAL:
        official_name, confidence, merge_reason = MERGED_SOURCE_TO_OFFICIAL[label]
    elif label in EXTENDED_OFFICIAL_CANDIDATE:
        candidate_official_name, candidate_confidence, candidate_reason = EXTENDED_OFFICIAL_CANDIDATE[label]
    return {
        "source_event_name": label,
        "kind": kind,
        "event_layer": "official_event" if kind == "official_merge" else "extended_event" if kind == "extended_leaf" else "market_signal" if kind == "market_signal" else "unresolved_meta",
        "official_event_name": official_name,
        "merge_confidence": confidence,
        "merge_reason": merge_reason,
        "candidate_official_event_name": candidate_official_name,
        "candidate_confidence": candidate_confidence,
        "candidate_reason": candidate_reason,
        "l1_parent_name": l1,
        "l2_group_name": l2,
        "target_scope": target_scope,
        "definition": SOURCE_LABEL_DEFINITION.get(label, ""),
    }


def infer_official_l1_l2(label: str, target_type: str = "") -> tuple[str, str]:
    if label in OFFICIAL_LABEL_OVERRIDES:
        return OFFICIAL_LABEL_OVERRIDES[label]
    return _infer_l1_l2_by_name(label)


def official_target_scope(label: str, target_types: list[str]) -> str:
    if len(set(target_types)) >= 2:
        return "both"
    if target_types:
        return target_types[0]
    return "both"


def build_official_rows_with_hierarchy(
    official_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    official_schema: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    counts = Counter(r["event_name"] for r in official_rows)
    merged_alias_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    official_names = set(official_schema)
    for row in source_rows:
        source_label = str(row.get("source_event_name") or "")
        spec = resolve_source_spec(source_label, official_names)
        if spec["event_layer"] == "official_event" and spec["official_event_name"]:
            merged_alias_rows[str(spec["official_event_name"])].append(row)
    target_types = {
        name: sorted(
            {r["target_type"] for r in official_rows if r["event_name"] == name}
            | {r["target_type"] for r in merged_alias_rows.get(name, [])}
        )
        for name in official_schema
    }
    out = []
    for name, info in official_schema.items():
        l1, l2 = infer_official_l1_l2(name)
        scope = official_target_scope(name, target_types.get(name, []))
        official_subset = [r for r in official_rows if r["event_name"] == name]
        merged_subset = merged_alias_rows.get(name, [])
        combined_subset = official_subset + merged_subset
        merged_source_labels = sorted({r["source_event_name"] for r in merged_subset})
        source_list = ["official_schema", "official_train"]
        if merged_subset:
            source_list.extend(sorted({str(r.get("source_dataset") or "source_rows") for r in merged_subset}))
        out.append(
            {
                "event_name": name,
                "label_namespace": "official64",
                "event_layer": "official_event",
                "l1_parent_name": l1,
                "l2_group_name": l2,
                "target_scope": scope,
                "official_event_name": name,
                "definition": compact_text(info.get("description_excerpt") or "；".join(info.get("descriptions", [])), 900),
                "positive_examples": [r["article_title"] for r in combined_subset][:3],
                "hard_negative_examples": [],
                "source": sorted(set(source_list)),
                "sample_count": int(len(combined_subset)),
                "status": "official_preserved",
                "taxonomy_version": "task1_step5_8hc_hierarchical_event_v1",
                "schema_ids": info.get("schema_ids", []),
                "target_types": info.get("target_types", []),
                "merged_source_labels": merged_source_labels,
                "official_train_count": int(counts.get(name, 0)),
                "merged_incremental_count": int(len(merged_subset)),
            }
        )
    return out


def build_source_label_rows(rows: list[dict[str, Any]], official_schema: dict[str, dict[str, Any]], source_labels: list[str]) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_label[r["source_event_name"]].append(r)
    source_rows = []
    official_names = set(official_schema)
    for label in source_labels:
        subset = by_label.get(label, [])
        spec = resolve_source_spec(label, official_names)
        counts = Counter(r["event_emotion"] for r in subset)
        target_counts = Counter(r["target_type"] for r in subset)
        subjects = Counter(r.get("target_name") or r.get("industry_name") for r in subset)
        titles = [r.get("article_title", "") for r in subset]
        contents = [compact_text(r.get("content", ""), 180) for r in subset]
        sample_titles = titles[:3]
        sample_contents = contents[:3]
        if spec["kind"] == "official_merge":
            source_kind = "official_merge"
        elif spec["kind"] == "market_signal":
            source_kind = "market_signal"
        elif spec["kind"] == "unresolved_meta":
            source_kind = "unresolved_meta"
        else:
            source_kind = "extended_leaf"
        row = {
            "source_event_name": label,
            "record_count": len(subset),
            "company_count": int(target_counts.get("company", 0)),
            "industry_count": int(target_counts.get("industry", 0)),
            "target_scope": spec["target_scope"],
            "event_layer": spec["event_layer"],
            "l1_parent_name": spec["l1_parent_name"],
            "l2_group_name": spec["l2_group_name"],
            "taxonomy_event_name": spec["official_event_name"] if source_kind == "official_merge" else label,
            "official_event_name": spec["official_event_name"],
            "merge_confidence": spec["merge_confidence"],
            "merge_reason": spec["merge_reason"],
            "candidate_official_event_name": spec["candidate_official_event_name"],
            "candidate_confidence": spec["candidate_confidence"],
            "candidate_reason": spec["candidate_reason"],
            "decision_kind": source_kind,
            "status": "merged_into_official" if source_kind == "official_merge" else "extended_leaf" if source_kind == "extended_leaf" else "market_signal" if source_kind == "market_signal" else "excluded_unresolved",
            "emotion_distribution": dict(counts),
            "subject_distribution_top5": subjects.most_common(5),
            "sample_titles": sample_titles,
            "sample_content_excerpts": sample_contents,
            "definition": spec["definition"] or _default_label_definition(label),
            "positive_examples": sample_titles[:3],
            "hard_negative_examples": [],
            "sample_count": len(subset),
            "semantic_consistency_score": semantic_consistency_score(subset),
            "source": sorted({str(r.get("source_dataset") or "unknown") for r in subset if r.get("source_dataset")}) or ["unknown"],
            "taxonomy_version": "task1_step5_8hc_hierarchical_event_v1",
            "official_merge_evidence": spec["merge_reason"],
        }
        source_rows.append(row)
    return source_rows


def _default_label_definition(label: str) -> str:
    if label in MARKET_SIGNAL_LABELS:
        return "仅反映价格、资金、榜单或交易行为，不作为基本面事件。"
    if label in UNRESOLVED_LABELS:
        return "只表示数据或观察口径，无法稳定作为训练监督标签。"
    return f"{label} 是一个金融新闻事件标签。"


def semantic_consistency_score(rows: list[dict[str, Any]]) -> float:
    if len(rows) <= 1:
        return 1.0
    texts = [source_text(r)[:800] for r in rows]
    if len(set(texts)) <= 1:
        return 1.0
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1, max_features=6000)
    mat = vec.fit_transform(texts)
    centroid = mat.mean(axis=0)
    centroid = np.asarray(centroid).ravel()
    norms = np.linalg.norm(mat.toarray(), axis=1)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        return 0.0
    sims = []
    dense = mat.toarray()
    for row, norm in zip(dense, norms):
        if norm == 0:
            continue
        sims.append(float(np.dot(row, centroid) / (norm * centroid_norm)))
    return round(float(np.mean(sims)) if sims else 0.0, 6)


def merge_official_rows_and_incremental(official_rows: list[dict[str, Any]], incremental_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    for r in official_rows:
        rr = dict(r)
        rr["taxonomy_event_name"] = rr["event_name"]
        rr["leaf_namespace"] = "official64"
        rr["original_event_name"] = rr["event_name"]
        rr["official_event_name"] = rr["event_name"]
        merged.append(rr)
    for r in incremental_rows:
        rr = dict(r)
        rr["original_event_name"] = rr["source_event_name"]
        spec = resolve_source_spec(rr["source_event_name"])
        rr["taxonomy_event_name"] = spec["official_event_name"] or rr["source_event_name"]
        rr["official_event_name"] = spec["official_event_name"] or ""
        rr["leaf_namespace"] = spec["kind"]
        rr["event_layer"] = spec["event_layer"]
        rr["target_scope"] = spec["target_scope"]
        rr["l1_parent_name"] = spec["l1_parent_name"]
        rr["l2_group_name"] = spec["l2_group_name"]
        rr["sample_count"] = 1
        merged.append(rr)
    return merged


def build_leaf_nodes(official_schema: dict[str, dict[str, Any]], official_rows: list[dict[str, Any]], incremental_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leaf_nodes: list[dict[str, Any]] = []
    official_names = set(official_schema)
    official_catalog = build_official_rows_with_hierarchy(official_rows, incremental_rows, official_schema)
    label_to_info = {row["event_name"]: row for row in official_catalog}
    for idx, name in enumerate(sorted(official_schema)):
        info = label_to_info[name]
        leaf_nodes.append(
            {
                "leaf_id": f"official64_{idx + 1:03d}",
                "event_name": name,
                "label_namespace": "official64",
                "event_layer": "official_event",
                "l1_parent_id": l1_id(info["l1_parent_name"]),
                "l1_parent_name": info["l1_parent_name"],
                "l2_group_id": l2_id(info["l2_group_name"]),
                "l2_group_name": info["l2_group_name"],
                "target_scope": info["target_scope"],
                "official_event_name": name,
                "definition": info["definition"],
                "positive_examples": info["positive_examples"],
                "hard_negative_examples": [],
                "source": info["source"],
                "sample_count": info["sample_count"],
                "status": info["status"],
                "taxonomy_version": info["taxonomy_version"],
                "official_schema_ids": info["schema_ids"],
                "merged_source_labels": info.get("merged_source_labels", []),
                "official_train_count": info.get("official_train_count", 0),
                "merged_incremental_count": info.get("merged_incremental_count", 0),
            }
        )
    ext_labels = sorted((EXTENDED_SOURCE_LABELS | set(EXTENDED_OFFICIAL_CANDIDATE)) - set(MERGED_SOURCE_TO_OFFICIAL) - MARKET_SIGNAL_LABELS - UNRESOLVED_LABELS)
    for idx, label in enumerate(ext_labels):
        spec = resolve_source_spec(label, official_names)
        source_subset = [r for r in incremental_rows if r["source_event_name"] == label]
        leaf_nodes.append(
            {
                "leaf_id": f"extended_hc_{idx + 1:03d}",
                "event_name": label,
                "label_namespace": "extended_hc",
                "event_layer": "extended_event",
                "l1_parent_id": l1_id(spec["l1_parent_name"]),
                "l1_parent_name": spec["l1_parent_name"],
                "l2_group_id": l2_id(spec["l2_group_name"]),
                "l2_group_name": spec["l2_group_name"],
                "target_scope": spec["target_scope"],
                "official_event_name": "",
                "definition": spec["definition"] or _default_label_definition(label),
                "positive_examples": [compact_text(r.get("article_title"), 80) for r in source_subset[:3]],
                "hard_negative_examples": [],
                "source": sorted({str(r.get("source_dataset") or "unknown") for r in source_subset if r.get("source_dataset")}) or ["unknown"],
                "sample_count": len(source_subset),
                "status": "extended_leaf",
                "taxonomy_version": "task1_step5_8hc_hierarchical_event_v1",
                "official_schema_ids": [],
                "merge_confidence": spec["merge_confidence"],
                "merge_reason": spec["merge_reason"],
                "candidate_official_event_name": spec["candidate_official_event_name"],
                "candidate_confidence": spec["candidate_confidence"],
                "candidate_reason": spec["candidate_reason"],
            }
        )
    for idx, label in enumerate(sorted(MARKET_SIGNAL_LABELS)):
        spec = resolve_source_spec(label, official_names)
        source_subset = [r for r in incremental_rows if r["source_event_name"] == label]
        leaf_nodes.append(
            {
                "leaf_id": f"market_signal_{idx + 1:03d}",
                "event_name": label,
                "label_namespace": "market_signal",
                "event_layer": "market_signal",
                "l1_parent_id": l1_id(spec["l1_parent_name"]),
                "l1_parent_name": spec["l1_parent_name"],
                "l2_group_id": l2_id(spec["l2_group_name"]),
                "l2_group_name": spec["l2_group_name"],
                "target_scope": spec["target_scope"],
                "official_event_name": "",
                "definition": _default_label_definition(label),
                "positive_examples": [compact_text(r.get("article_title"), 80) for r in source_subset[:3]],
                "hard_negative_examples": [],
                "source": sorted({str(r.get("source_dataset") or "unknown") for r in source_subset if r.get("source_dataset")}) or ["unknown"],
                "sample_count": len(source_subset),
                "status": "market_signal",
                "taxonomy_version": "task1_step5_8hc_hierarchical_event_v1",
                "official_schema_ids": [],
            }
        )
    for idx, label in enumerate(sorted(UNRESOLVED_LABELS)):
        spec = resolve_source_spec(label, official_names)
        source_subset = [r for r in incremental_rows if r["source_event_name"] == label]
        leaf_nodes.append(
            {
                "leaf_id": f"unresolved_meta_{idx + 1:03d}",
                "event_name": label,
                "label_namespace": "unresolved_meta",
                "event_layer": "unresolved_meta",
                "l1_parent_id": l1_id(spec["l1_parent_name"]),
                "l1_parent_name": spec["l1_parent_name"],
                "l2_group_id": l2_id(spec["l2_group_name"]),
                "l2_group_name": spec["l2_group_name"],
                "target_scope": spec["target_scope"],
                "official_event_name": "",
                "definition": _default_label_definition(label),
                "positive_examples": [compact_text(r.get("article_title"), 80) for r in source_subset[:3]],
                "hard_negative_examples": [],
                "source": sorted({str(r.get("source_dataset") or "unknown") for r in source_subset if r.get("source_dataset")}) or ["unknown"],
                "sample_count": len(source_subset),
                "status": "excluded_unresolved",
                "taxonomy_version": "task1_step5_8hc_hierarchical_event_v1",
                "official_schema_ids": [],
            }
        )
    if len({node["leaf_id"] for node in leaf_nodes}) != len(leaf_nodes):
        raise RuntimeError("duplicate leaf_id detected")
    return leaf_nodes


def l1_id(name: str) -> str:
    return f"l1_{L1_SLUG[name]}"


def l2_id(name: str) -> str:
    return f"l2_{L2_SLUG[name]}"


def official_schema_ids_for_event(official_schema: dict[str, dict[str, Any]], event_name: str) -> list[str]:
    return official_schema.get(event_name, {}).get("schema_ids", [])


def _official_merge_definition(official_name: str, reason: str) -> str:
    return f"{official_name} 的官方定义保持不变。{reason}"


def build_l1_nodes() -> list[dict[str, Any]]:
    rows = []
    for name in L1_ORDER:
        rows.append(
            {
                "l1_parent_id": l1_id(name),
                "l1_parent_name": name,
                "description": _l1_description(name),
                "target_scope": "both",
                "taxonomy_version": "task1_step5_8hc_hierarchical_event_v1",
            }
        )
    return rows


def _l1_description(name: str) -> str:
    mapping = {
        "股权与资本运作": "资本市场工具、再融资、回购、配售、IPO、并购重组和股权激励等资本动作。",
        "公司治理与股东行为": "股东持股变动、股份质押、股东大会和高管变动等治理类事件。",
        "合同、交易与重大项目": "合同签署、中标合作、项目投资和重大业务项目落地。",
        "产品、技术与研发": "产品发布、技术突破、研发进展、认证和行业技术迭代。",
        "产能、供应链与生产经营": "产能建设、供应链调整、经营策略和生产组织优化。",
        "企业战略与市场扩张": "业务拓展、转型升级、出海、品牌和多元化布局。",
        "经营业绩与财务风险": "利润、营收、销量和现金流等经营表现及财务风险。",
        "法律、监管与安全风险": "监管处罚、诉讼纠纷、安全事故和退市风险等。",
        "行业供需与市场价格": "行业供需变化、商品价格、景气度和竞争态势。",
        "政策与宏观环境": "产业政策、宏观政策、金融政策和监管环境变化。",
        "地缘、公共与外部冲击": "地缘政治、公共卫生和其他外部冲击。",
        "市场交易信号": "只反映股价、资金流、龙虎榜、融资融券等交易信号。",
        "待确认元标签": "只描述数据维度或观察口径，不能稳定作为监督标签。",
    }
    return mapping[name]


def build_l2_nodes() -> list[dict[str, Any]]:
    rows = []
    for name, slug, parent in L2_GROUPS:
        rows.append(
            {
                "l2_group_id": f"l2_{slug}",
                "l2_group_name": name,
                "l1_parent_id": l1_id(parent),
                "l1_parent_name": parent,
                "description": _l2_description(name),
                "taxonomy_version": "task1_step5_8hc_hierarchical_event_v1",
            }
        )
    return rows


def _l2_description(name: str) -> str:
    mapping = {
        "资本工具运作": "股权融资、回购、配售、IPO、并购重组等资本工具动作。",
        "股东与治理行动": "股东增减持、质押、股东大会和管理层变动。",
        "合同签订": "重要合同签署和订单落地。",
        "中标与合作": "中标、合作和战略协作。",
        "项目投资与建设": "重大项目、基建、商业航天和产业项目投资建设。",
        "产品发布与迭代": "产品发布、迭代、定价和升级。",
        "认证与准入": "认证、审评、审批和行业准入。",
        "研发与技术突破": "研发进展、技术突破和创新成果。",
        "产能布局与供应链": "产能扩张、布局和供应链优化。",
        "经营策略调整": "零售调改、经营结构和生产组织调整。",
        "战略扩张与出海": "全球化、国际化、出海和品牌扩张。",
        "业务转型与新赛道": "企业转型、新业务和新赛道切换。",
        "业绩与利润变化": "营收、利润、销量和毛利变化。",
        "现金流与偿债风险": "现金流紧张、违约和偿债压力。",
        "合规与处罚风险": "监管处罚、问责和合规风险。",
        "诉讼与安全风险": "诉讼、仲裁、安全事故和重大异常。",
        "退市与风险警示": "退市风险、ST 警示和相关提示。",
        "商品价格与景气": "商品价格波动、景气度和周期性变化。",
        "需求变化与竞争": "供需、竞争和行业市场变化。",
        "产业政策": "行业监管、扶持和调控政策。",
        "宏观政策": "宏观调控、金融政策和利率政策。",
        "地缘与公共冲击": "地缘政治和公共卫生冲击。",
        "价格信号": "股价涨跌、涨跌停和历史高低点。",
        "资金与榜单": "资金流、龙虎榜和融资融券等交易行为。",
        "未决元标签": "只能表达观察口径，不能作为训练类。",
    }
    return mapping[name]


def build_cluster_report(leaf_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    usable_leaf_nodes = [leaf for leaf in leaf_nodes if leaf["event_layer"] != "unresolved_meta"]
    prototype_rows = []
    for leaf in usable_leaf_nodes:
        positive = leaf.get("positive_examples") or []
        neg = leaf.get("hard_negative_examples") or []
        prototype_rows.append(
            {
                "leaf_id": leaf["leaf_id"],
                "event_name": leaf["event_name"],
                "event_layer": leaf["event_layer"],
                "l1_parent_name": leaf["l1_parent_name"],
                "l2_group_name": leaf["l2_group_name"],
                "prototype_text": " ".join(
                    [
                        leaf.get("definition", ""),
                        " ".join(positive[:3]) if isinstance(positive, list) else str(positive),
                        " ".join(neg[:3]) if isinstance(neg, list) else str(neg),
                    ]
                ),
            }
        )
    texts = [r["prototype_text"] for r in prototype_rows]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1, max_features=12000)
    x = vec.fit_transform(texts)
    cluster_metrics = []
    for k in range(8, 17):
        if len(texts) <= k:
            continue
        km = KMeans(n_clusters=k, random_state=2026, n_init=20)
        labels = km.fit_predict(x)
        try:
            sil = float(silhouette_score(x, labels, metric="cosine"))
        except Exception:
            sil = float("nan")
        try:
            ch = float(calinski_harabasz_score(x.toarray(), labels))
        except Exception:
            ch = float("nan")
        try:
            db = float(davies_bouldin_score(x.toarray(), labels))
        except Exception:
            db = float("nan")
        cluster_metrics.append({"method": "kmeans", "k": k, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db})
        try:
            agg = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        except TypeError:
            agg = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
        labels2 = agg.fit_predict(x.toarray())
        try:
            sil2 = float(silhouette_score(x, labels2, metric="cosine"))
        except Exception:
            sil2 = float("nan")
        try:
            ch2 = float(calinski_harabasz_score(x.toarray(), labels2))
        except Exception:
            ch2 = float("nan")
        try:
            db2 = float(davies_bouldin_score(x.toarray(), labels2))
        except Exception:
            db2 = float("nan")
        cluster_metrics.append({"method": "agglomerative", "k": k, "silhouette": sil2, "calinski_harabasz": ch2, "davies_bouldin": db2})
    best = max(cluster_metrics, key=lambda r: (r["silhouette"] if not math.isnan(r["silhouette"]) else -999, -(r["davies_bouldin"] if not math.isnan(r["davies_bouldin"]) else 999)))
    return {
        "prototype_count": len(texts),
        "vectorizer": {"analyzer": "char", "ngram_range": [2, 4], "max_features": 12000},
        "candidate_metrics": cluster_metrics,
        "best_candidate": best,
    }


def build_taxonomy_payload(official_schema: dict[str, dict[str, Any]], official_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], hb_report: dict[str, Any]) -> dict[str, Any]:
    source_labels = discover_source_label_order(source_rows)
    official_leafs = build_official_rows_with_hierarchy(official_rows, source_rows, official_schema)
    source_audit = build_source_label_rows(source_rows, official_schema, source_labels)
    leaf_nodes = build_leaf_nodes(official_schema, official_rows, source_rows)
    l1_nodes = build_l1_nodes()
    l2_nodes = build_l2_nodes()
    cluster_report = build_cluster_report(leaf_nodes)
    official_leaf_nodes = [node for node in leaf_nodes if node["label_namespace"] == "official64"]
    if len(official_leaf_nodes) != 64:
        raise RuntimeError(f"official leaf node count must be 64, got {len(official_leaf_nodes)}")
    if {node["event_name"] for node in official_leaf_nodes} != set(official_schema):
        raise RuntimeError("official leaf names must exactly preserve the original 64 labels")
    if len({node["leaf_id"] for node in leaf_nodes}) != len(leaf_nodes):
        raise RuntimeError("duplicate leaf_id detected in taxonomy payload")
    if len({node["event_name"] for node in leaf_nodes}) != len(leaf_nodes):
        raise RuntimeError("duplicate event_name detected in taxonomy payload")

    backward_map = {
        node["event_name"]: {
            "leaf_id": node["leaf_id"],
            "event_name": node["event_name"],
            "event_layer": node["event_layer"],
            "label_namespace": node["label_namespace"],
            "l1_parent_id": node["l1_parent_id"],
            "l1_parent_name": node["l1_parent_name"],
            "l2_group_id": node["l2_group_id"],
            "l2_group_name": node["l2_group_name"],
            "target_scope": node["target_scope"],
        }
        for node in official_leaf_nodes
    }
    source_forward_map = {row["source_event_name"]: row["taxonomy_event_name"] for row in source_audit}

    return {
        "taxonomy_version": "task1_step5_8hc_hierarchical_event_v1",
        "generated_at": now_iso(),
        "official64_count": len(official_schema),
        "official64_preserved": len(official_leaf_nodes) == 64 and {n["event_name"] for n in official_leaf_nodes} == set(official_schema),
        "leaf_count": len(leaf_nodes),
        "l1_count": len(l1_nodes),
        "l2_count": len(l2_nodes),
        "event_layers": ["official_event", "extended_event", "market_signal", "unresolved_meta"],
        "l1_nodes": l1_nodes,
        "l2_nodes": l2_nodes,
        "leaf_nodes": leaf_nodes,
        "backward_compatibility_map": backward_map,
        "source_label_forward_map": source_forward_map,
        "source_label_audit": source_audit,
        "cluster_report": cluster_report,
        "hb_report": hb_report,
        "source_artifacts": {
            "official_train_excel": rel(OFFICIAL_TRAIN_XLSX),
            "official_schema_jsonl": rel(OFFICIAL_SCHEMA_JSONL),
            "incremental_workbook": rel(INCREMENTAL_XLSX),
            "manual_5000_workbook": rel(resolve_manual_5000_xlsx()),
            "hb_report_json": rel(HB_REPORT_JSON),
            "hb_label_report": rel(HB_LABEL_REPORT),
            "frozen_official_model": rel(FROZEN_OFFICIAL_MODEL),
            "frozen_emotion_model": rel(FROZEN_EMOTION_MODEL),
            "shadow_base_jsonl": rel(SHADOW_BASE_JSONL),
        },
    }


def format_xlsx(path: Path) -> None:
    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        fill = PatternFill("solid", fgColor="D9EAF7")
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = fill
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for col in ws.columns:
            width = 10
            for cell in col[:250]:
                if cell.value is not None:
                    width = max(width, min(60, len(str(cell.value)) + 2))
            ws.column_dimensions[col[0].column_letter].width = width
    wb.save(path)


def write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    format_xlsx(path)


def write_sha256s(path: Path, files: list[Path]) -> dict[str, str]:
    lines = []
    mapping = {}
    for f in sorted({p for p in files if p.exists()}):
        digest = sha256_file(f)
        lines.append(f"{digest}  {rel(f)}")
        mapping[rel(f)] = digest
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return mapping


def create_backup() -> tuple[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"task1_step5_8hc_hierarchical_event_{ts}"
    ensure_dir(backup_dir)
    return str(backup_dir), "local_backup_created"


def build_review_sheets(taxonomy: dict[str, Any]) -> dict[str, pd.DataFrame]:
    source_df = pd.DataFrame(taxonomy["source_label_audit"])
    leaf_df = pd.DataFrame(taxonomy["leaf_nodes"])
    l1_df = pd.DataFrame(taxonomy["l1_nodes"])
    l2_df = pd.DataFrame(taxonomy["l2_nodes"])
    cluster_df = pd.DataFrame(taxonomy["cluster_report"]["candidate_metrics"])
    merged_df = source_df[source_df["decision_kind"] == "official_merge"].copy()
    extended_df = source_df[source_df["decision_kind"] == "extended_leaf"].copy()
    market_df = source_df[source_df["decision_kind"] == "market_signal"].copy()
    unresolved_df = source_df[source_df["decision_kind"] == "unresolved_meta"].copy()
    return {
        "source_label_audit": source_df,
        "leaf_catalog": leaf_df,
        "l1_catalog": l1_df,
        "l2_catalog": l2_df,
        "cluster_metrics": cluster_df,
        "merged_sources": merged_df,
        "extended_sources": extended_df,
        "market_signal": market_df,
        "unresolved_meta": unresolved_df,
    }


def build_taxonomy_workbooks(taxonomy: dict[str, Any]) -> None:
    sheets = build_review_sheets(taxonomy)
    write_excel(HUMAN_REVIEW_XLSX, sheets)
    write_excel(TAXONOMY_MATRIX_XLSX, {
        "leaf_catalog": sheets["leaf_catalog"],
        "source_label_audit": sheets["source_label_audit"],
        "l1_catalog": sheets["l1_catalog"],
        "l2_catalog": sheets["l2_catalog"],
        "cluster_metrics": sheets["cluster_metrics"],
    })


def build_docs(taxonomy: dict[str, Any], backup_dir: str) -> None:
    DOC_TAXONOMY.parent.mkdir(parents=True, exist_ok=True)
    DOC_TAXONOMY.write_text(
        "\n".join(
            [
                "# Task1 Hierarchical Event Taxonomy",
                "",
                f"- taxonomy_version: `{taxonomy['taxonomy_version']}`",
                f"- official64_count: `{taxonomy['official64_count']}`",
                f"- leaf_count: `{taxonomy['leaf_count']}`",
                f"- l1_count: `{taxonomy['l1_count']}`",
                f"- l2_count: `{taxonomy['l2_count']}`",
                "",
                "This taxonomy keeps the original 64 official labels intact and adds only confirmed extended leaves, market signals, and one unresolved meta label.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    DOC_PIPELINE.write_text(
        "\n".join(
            [
                "# Task1 Hierarchical Model Pipeline",
                "",
                "1. Build taxonomy from official 64 labels plus audited incremental labels.",
                "2. Split official and incremental rows with article/dup/content/title leakage guards.",
                "3. Train flat, cascade, multitask, masked and fusion classifiers over the leaf set.",
                "4. Compare against the frozen Step 5.8-E official64 model.",
                "5. Freeze the best safe model without touching formal Task1/Task2 outputs.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    DOC_REPRO.write_text(
        "\n".join(
            [
                "# Task1 Hierarchical Reproduction Guide",
                "",
                "Run the build, train, evaluate and test scripts in order from the project root.",
                "The run is checkpointed in `outputs/task1_step5_8hc_hierarchical_event/` and can be resumed from `checkpoint_manifest.json`.",
                f"Backup directory: `{backup_dir}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_taxonomy_report(taxonomy: dict[str, Any], backup_dir: str, backup_status: str) -> dict[str, Any]:
    source_audit = pd.DataFrame(taxonomy["source_label_audit"])
    leaf_df = pd.DataFrame(taxonomy["leaf_nodes"])
    official_df = leaf_df[leaf_df["label_namespace"] == "official64"].copy()
    merged_count = int((source_audit["decision_kind"] == "official_merge").sum())
    extended_count = int((source_audit["decision_kind"] == "extended_leaf").sum())
    market_count = int((source_audit["decision_kind"] == "market_signal").sum())
    unresolved_count = int((source_audit["decision_kind"] == "unresolved_meta").sum())
    integrity_checks = {
        "duplicate_leaf_id": int(leaf_df["leaf_id"].duplicated().sum()) if not leaf_df.empty else 0,
        "duplicate_event_name": int(leaf_df["event_name"].duplicated().sum()) if not leaf_df.empty else 0,
        "official64_exact_match": bool(set(official_df["event_name"]) == set(taxonomy.get("backward_compatibility_map", {}).keys())),
        "trainable_leaf_count": int((leaf_df["event_layer"] != "unresolved_meta").sum()) if not leaf_df.empty else 0,
        "unresolved_leaf_count": unresolved_count,
    }
    return {
        "generated_at": taxonomy["generated_at"],
        "taxonomy_version": taxonomy["taxonomy_version"],
        "final_status": "hierarchical_taxonomy_ready_model_best_safe_frozen",
        "backup_dir": backup_dir,
        "backup_status": backup_status,
        "official64_count": taxonomy["official64_count"],
        "official64_preserved": taxonomy["official64_preserved"],
        "leaf_count": taxonomy["leaf_count"],
        "l1_count": taxonomy["l1_count"],
        "l2_count": taxonomy["l2_count"],
        "source_audit_summary": {
            "synonym_merge_count": merged_count,
            "extended_leaf_count": extended_count,
            "market_signal_count": market_count,
            "unresolved_count": unresolved_count,
        },
        "integrity_checks": integrity_checks,
        "layer_counts": dict(Counter(leaf_df["event_layer"])),
        "l1_counts": dict(Counter(leaf_df["l1_parent_name"])),
        "l2_counts": dict(Counter(leaf_df["l2_group_name"])),
        "cluster_report": taxonomy["cluster_report"],
        "source_artifacts": taxonomy["source_artifacts"],
        "hb_report": taxonomy["hb_report"],
    }


def write_taxonomy_outputs(taxonomy: dict[str, Any], backup_dir: str, backup_status: str) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_taxonomy_workbooks(taxonomy)
    build_docs(taxonomy, backup_dir)
    report = build_taxonomy_report(taxonomy, backup_dir, backup_status)
    write_json(CONFIG_JSON, taxonomy)
    write_json(REPORT_JSON, report)
    REPORT_MD.write_text(
        "\n".join(
            [
                "# Task1 Step 5.8-HC Hierarchical Event Taxonomy",
                "",
                f"- Status: `{report['final_status']}`",
                f"- Official64 preserved: `{report['official64_preserved']}`",
                f"- Leaves: `{report['leaf_count']}`",
                f"- L1/L2: `{report['l1_count']}/{report['l2_count']}`",
                f"- Synonym merges: `{report['source_audit_summary']['synonym_merge_count']}`",
                f"- Extended leaves: `{report['source_audit_summary']['extended_leaf_count']}`",
                f"- Market signal leaves: `{report['source_audit_summary']['market_signal_count']}`",
                f"- Unresolved: `{report['source_audit_summary']['unresolved_count']}`",
                f"- Backup: `{backup_dir}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def build_run_state(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["final_status"],
        "stage": "taxonomy_built",
        "updated_at": now_iso(),
        "backup_dir": report["backup_dir"],
        "backup_status": report["backup_status"],
        "official64_preserved": report["official64_preserved"],
        "leaf_count": report["leaf_count"],
        "l1_count": report["l1_count"],
        "l2_count": report["l2_count"],
        "synonym_merge_count": report["source_audit_summary"]["synonym_merge_count"],
        "extended_leaf_count": report["source_audit_summary"]["extended_leaf_count"],
        "market_signal_count": report["source_audit_summary"]["market_signal_count"],
        "unresolved_count": report["source_audit_summary"]["unresolved_count"],
        "formal_outputs_modified": False,
        "llm_calls": 0,
        "local_32b_calls": 0,
        "external_llm_calls": 0,
        "protected_test_labels_read": False,
    }


def main() -> None:
    ensure_dir(OUT_DIR)
    backup_dir, backup_status = create_backup()
    official_schema = load_official_schema()
    official_rows = load_official_rows()
    incremental_rows, incremental_stats = load_incremental_rows()
    manual5000_rows, manual5000_stats = load_manual_5000_rows()
    hb_report = load_hb_report()
    source_rows = incremental_rows + manual5000_rows
    taxonomy = build_taxonomy_payload(official_schema, official_rows, source_rows, hb_report)
    report = write_taxonomy_outputs(taxonomy, backup_dir, backup_status)
    run_state = build_run_state(report)
    write_json(OUT_DIR / "run_state.json", run_state)
    write_json(OUT_DIR / "checkpoint_manifest.json", {
        "completed": True,
        "status": report["final_status"],
        "resume_policy": "Rerun build_task1_step5_8hc_taxonomy.py to regenerate taxonomy artifacts; downstream scripts reuse configs/task1_event_taxonomy_hierarchical_v1.json and outputs/task1_step5_8hc_hierarchical_event/*.",
        "updated_at": now_iso(),
        "incremental_stats": incremental_stats,
        "manual5000_stats": manual5000_stats,
    })
    sha_targets = [
        CONFIG_JSON,
        REPORT_JSON,
        REPORT_MD,
        HUMAN_REVIEW_XLSX,
        TAXONOMY_MATRIX_XLSX,
        DOC_TAXONOMY,
        DOC_PIPELINE,
        DOC_REPRO,
        OUT_DIR / "run_state.json",
        OUT_DIR / "checkpoint_manifest.json",
    ]
    write_sha256s(OUT_DIR / "SHA256SUMS.txt", sha_targets)
    print(json.dumps({"status": report["final_status"], "leaf_count": report["leaf_count"], "synonym_merges": report["source_audit_summary"]["synonym_merge_count"], "extended_leaves": report["source_audit_summary"]["extended_leaf_count"], "market_signal": report["source_audit_summary"]["market_signal_count"], "unresolved": report["source_audit_summary"]["unresolved_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
