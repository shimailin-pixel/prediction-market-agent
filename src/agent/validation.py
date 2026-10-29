"""有限的离线证据筛查，不等同于完整事实核验。"""
import json
import re
from decimal import Decimal

from .prompt import build_evidence


def number_tokens(text):
    # 保留数值、百分号及量级单位，防止把 85% 与 85 美元视为同一证据。
    out = set()
    for value, unit in re.findall(r"(?<![\d.])([+-]?\d[\d,]*(?:\.\d+)?)\s*(%|[KMBkmb](?![a-zA-Z])|万|亿)?", text):
        amount = Decimal(value.replace(",", ""))
        factors = {"K":1000, "M":1000000, "B":1000000000, "万":10000, "亿":100000000}
        if unit.upper() in factors:
            amount *= factors[unit.upper()]
            unit = ""
        out.add((amount, unit or ""))
    return out


def validate_evidence(data, anomaly):
    """返回可读拒绝原因；不修改或自动补全模型的结论。"""
    narrative = json.dumps({k: v for k, v in data.items() if k in {
        "event_summary", "impact_on_a_share", "impact_direction", "trade_bias", "rationale", "risk_note"
    }}, ensure_ascii=False)
    m = anomaly.market
    source = (f"{anomaly.question} 当前概率 {anomaly.current_probability*100:.1f}% "
              + build_evidence(anomaly.details, anomaly.anomaly_type)
              + (m.description if m else ""))
    allowed = number_tokens(source)
    # “下降21.5点”引用的是带符号变化 -21.5 的绝对幅度。
    allowed |= {(abs(value), unit) for value, unit in allowed if value < 0}
    # 标题里的 October 与中文“10月”是同一日期信息。
    months = "january february march april may june july august september october november december".split()
    for index, month in enumerate(months, 1):
        if re.search(r"\b(?:" + month + "|" + month[:3] + r")\b", source, re.I):
            allowed.add((Decimal(index), ""))
    issues = []
    series = anomaly.details.get("threshold_series", "")
    if ("超出观测范围" in series or "不可估计" in series) and re.search(r"中位数(?:约|为|是|[:：])?\s*[$]?\d", narrative):
        issues.append("阶梯不足以估计精确中位数，解读却给出点估计")
    if number_tokens(narrative) - allowed:
        issues.append("解读包含输入证据未提供的数字或单位")
    if re.search(r"准确率|胜率|历史校准|回测|预测准确|calibrat|accuracy|win rate", narrative, re.I):
        issues.append("输入未提供历史校准或预测准确率证据")
    tags = set(anomaly.anomaly_type.split("+"))
    has_momentum = ("prob_spike" in tags and anomaly.details.get("change_pts") is not None) or any(
        anomaly.details.get(k) is not None for k in ("prob_change_24h_pts", "prob_change_7d_pts"))
    if not has_momentum and re.search(r"升至|升到|降至|降到|骤升|骤降|概率[^。；]{0,12}(上涨|上升|下降|上调|下调)|预期[^。；]{0,10}(强化|收紧|凝聚)", narrative):
        issues.append("缺少概率变化证据，却声称概率或预期发生变化")
    if re.search(r"资金[加速持续]*流入|资金[加速持续]*涌入|巨鲸涌入|净买入|净流入", narrative):
        issues.append("成交占比或持仓样本不能证明净资金流入")
    return issues
