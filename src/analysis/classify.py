"""事件分类与资产映射。

把一句市场问题映射到「事件类型」+「问题形态」，是后续信号解读与资产建议的基础。
沿用 polymarket_monitor 验证过的规则：
- 关键词/类型用**词边界正则**匹配，消除 toward→war 之类子串误伤；
- 区分 threshold（价格水平，如 above $110）与 event（会不会发生），二者动作不同；
- 事件类型优先级：机制类(ceasefire/war/nuclear) > 宏观类(interest_rate/inflation) >
  资产类(oil/crypto) > 地理类(china/russia)——因为交易逻辑由事件性质主导。
"""

from __future__ import annotations

import re
from typing import Dict, List

from ..models import Market

# 事件类型 → 资产动作映射。high/low/mid 针对事件型；level_high/level_low 针对阈值型。
EVENT_ASSET_MAP: Dict[str, Dict[str, Dict[str, str]]] = {
    "war": {
        "high": {"action": "做多黄金/原油，做空科技股", "reason": "冲突升级，避险资产上涨"},
        "low": {"action": "平仓避险头寸，买入风险资产", "reason": "冲突风险消退，情绪修复"},
        "mid": {"action": "方向性价比低，优先用期权表达波动率", "reason": "概率僵持 40-60%"},
    },
    "ceasefire": {
        "high": {"action": "平仓原油多头，买入被制裁国资产", "reason": "停火利好风险资产"},
        "low": {"action": "维持避险配置", "reason": "冲突延续概率高"},
        "mid": {"action": "观望，事件型波动率更适合当前定价", "reason": "停火与否未定"},
    },
    "nuclear": {
        "high": {"action": "做多黄金，做空风险资产", "reason": "核风险溢价上升"},
        "low": {"action": "减仓避险配置", "reason": "核风险消退"},
        "mid": {"action": "以波动率工具表达尾部风险", "reason": "概率未极端化"},
    },
    "tariff": {
        "high": {"action": "回避出口型企业，关注内需板块", "reason": "关税冲击出口链"},
        "low": {"action": "关注出口复苏机会", "reason": "贸易环境改善"},
        "mid": {"action": "出口链暂不建仓，等方向明确", "reason": "关税政策存在反复"},
    },
    "sanctions": {
        "high": {"action": "回避涉制裁产业链", "reason": "制裁收紧"},
        "low": {"action": "关注被制裁国资产修复", "reason": "制裁放松预期"},
        "mid": {"action": "等待制裁细则落地", "reason": "政策不确定"},
    },
    "recession": {
        "high": {"action": "增配国债和黄金，减仓周期股", "reason": "衰退避险"},
        "low": {"action": "增配成长股和周期股", "reason": "经济前景向好"},
        "mid": {"action": "均衡配置，周期与成长对半", "reason": "衰退与否分歧大"},
    },
    "interest_rate": {
        "high": {"action": "利率上行，配置银行股，压制高久期资产", "reason": "加息利好银行净息差"},
        "low": {"action": "降息预期，关注成长股", "reason": "低利率利好高估值标的"},
        "mid": {"action": "利率路径不清，减少久期暴露", "reason": "政策路径分歧"},
    },
    "inflation": {
        "high": {"action": "增配上游资源，回避下游制造", "reason": "通胀预期上行"},
        "low": {"action": "关注成本改善的下游行业", "reason": "通胀回落"},
        "mid": {"action": "关注实际利率敏感资产", "reason": "数据分歧"},
    },
    "oil": {
        "high": {"action": "做多原油及相关产业链", "reason": "油价上行概率高"},
        "low": {"action": "做空原油，航空化工成本端受益", "reason": "油价下行概率高"},
        "mid": {"action": "卖出宽波幅优于方向性押注", "reason": "油价区间震荡"},
    },
    "crypto": {
        "high": {"action": "做多比特币及相关贝塔资产", "reason": "加密上行概率高"},
        "low": {"action": "减仓加密敞口", "reason": "下行概率高"},
        "mid": {"action": "双向期权优于方向性持仓", "reason": "方向未定"},
    },
    "equity_index": {
        "high": {"action": "指数中枢上移，增配宽基", "reason": "阈值上方概率高"},
        "low": {"action": "指数中枢下移，降低权益敞口", "reason": "阈值下方概率高"},
        "mid": {"action": "减少方向性暴露", "reason": "指数区间震荡"},
    },
    "iran": {
        "high": {"action": "做多原油与黄金，回避伊朗相关权益", "reason": "伊朗局势升温"},
        "low": {"action": "减仓避险，关注原油回落风险", "reason": "局势缓和"},
        "mid": {"action": "原油与黄金双向波动预期上升", "reason": "局势不明"},
    },
    "mideast": {
        "high": {"action": "做多黄金原油，回避区域风险资产", "reason": "中东局势升级"},
        "low": {"action": "减仓避险配置", "reason": "中东局势缓和"},
        "mid": {"action": "以黄金期权表达尾部风险", "reason": "局势不明"},
    },
    "russia": {
        "high": {"action": "做多黄金与能源，规避涉俄产业链", "reason": "冲突升级"},
        "low": {"action": "关注被制裁俄资产修复", "reason": "缓和预期"},
        "mid": {"action": "等待局势明朗", "reason": "分歧较大"},
    },
    "china": {
        "high": {"action": "关注内需与自主可控，回避出口链", "reason": "政策收紧或摩擦升级"},
        "low": {"action": "关注外贸与顺周期修复", "reason": "关系缓和预期"},
        "mid": {"action": "政策预期不明，暂不押注方向", "reason": "分歧大"},
    },
}

# 类型识别正则（词边界）
TYPE_PATTERNS: Dict[str, str] = {
    "nuclear_agreement": r"\b(nuclear (?:deal|agreement|talks)|denuclearization|jcpoa)\b",
    "ceasefire": r"\b(ceasefire|cease-fire|truce|peace deal|peace agreement|armistice)\b",
    "nuclear": r"\b(nuclear|warhead|atomic|icbm|weapons of mass destruction)\b",
    "war": r"\b(war|wars|invasion|invade|invading|strike[sd]?|airstrike|offensive|missile)\b",
    "tariff": r"\b(tariff|tariffs|customs duty|trade war|import tax)\b",
    "sanctions": r"\b(sanction|sanctions|embargo|blacklist)\b",
    "interest_rate": r"\b(fed|fomc|federal reserve|interest rates?|rate cut|rate hike|policy rate|monetary policy|federal funds)\b",
    "recession": r"\b(recession|recessionary|contraction|downturn)\b",
    "inflation": r"\b(inflation|cpi|ppi|price index|deflation)\b",
    "oil": r"\b(oil|crude|brent|wti|opec|hormuz)\b",
    "crypto": r"\b(bitcoin|btc|ethereum|eth|crypto)\b",
    "equity_index": r"\b(s&p ?500|nasdaq|dow jones|stock market|equities)\b",
    "iran": r"\b(iran|iranian|tehran|irgc)\b",
    "mideast": r"\b(israel|israeli|gaza|hamas|hezbollah|houthi|lebanon|syria)\b",
    "russia": r"\b(russia|russian|ukraine|ukrainian|kremlin|moscow|kyiv)\b",
    "china": r"\b(china|chinese|beijing|taiwan|taipei)\b",
}

# 优先级：越靠前越具体，用于从多命中里选主类型
TYPE_PRIORITY: List[str] = [
    "nuclear_agreement", "ceasefire", "nuclear", "war", "tariff", "sanctions",
    "interest_rate", "recession", "inflation",
    "oil", "crypto", "equity_index",
    "iran", "mideast", "russia", "china",
]

LEVEL_PATTERNS = re.compile(
    r"\b(above|below|over|under|greater than|less than|more than|at least|"
    r"hit|reach|exceed|top|higher than|lower than|price of)\b",
    re.IGNORECASE,
)

# 检索关键词 → 类型（用于关键词兜底分类）
KEYWORD_TO_TYPE: Dict[str, str] = {
    "war": "war", "ceasefire": "ceasefire", "sanctions": "sanctions", "sanction": "sanctions",
    "tariff": "tariff", "recession": "recession", "interest rate": "interest_rate",
    "fed": "interest_rate", "rate cut": "interest_rate", "oil": "oil",
    "iran": "iran", "israel": "mideast", "gaza": "mideast",
    "russia": "russia", "ukraine": "russia", "china": "china", "taiwan": "china",
}


def types_from_question(question: str) -> List[str]:
    if not question:
        return []
    return [t for t, pat in TYPE_PATTERNS.items() if re.search(pat, question, re.IGNORECASE)]


def type_from_keyword(keyword: str) -> str:
    low = (keyword or "").lower().strip()
    if low in KEYWORD_TO_TYPE:
        return KEYWORD_TO_TYPE[low]
    for token, t in KEYWORD_TO_TYPE.items():
        if token in low:
            return t
    return ""


def classify_question_shape(question: str) -> str:
    if not question:
        return "unknown"
    if LEVEL_PATTERNS.search(question):
        return "threshold"
    if re.search(r"\b(will|does|do|is|are|has|have|can|would)\b", question, re.IGNORECASE):
        return "event"
    return "unknown"


def classify_signal_strength(yes_pct: float) -> str:
    if yes_pct >= 80:
        return "极强确定性"
    if yes_pct >= 60:
        return "较强倾向"
    if yes_pct >= 40:
        return "不确定（波动率机会）"
    if yes_pct >= 20:
        return "较强否定"
    return "极强否定"


def annotate(market: Market, keyword: str = "") -> Market:
    """就地为 Market 填充 event_types / primary_type / question_shape。"""
    text_types = types_from_question(market.question)
    kw_type = type_from_keyword(keyword) if keyword else ""
    types = text_types or ([kw_type] if kw_type else [])
    market.event_types = types
    market.primary_type = next((t for t in TYPE_PRIORITY if t in types), types[0] if types else "")
    market.question_shape = classify_question_shape(market.question)
    return market
