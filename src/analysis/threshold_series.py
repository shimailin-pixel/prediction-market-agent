"""阈值系列市场（Kalshi Fed 利率阶梯等）→ 隐含分布摘要。

同一事件（event_key 相同）的多个阈值合约，其 YES 概率构成离散累积分布：
above-x 型阶梯里 P(上限 ≥ x) 随 x 单调下降，把台阶连起来就是市场隐含的
利率分布——中位数即分布中位，两端概率即市场给尾部事件定的价。

逐行展示 8 个合约信息密度极低；折成一行摘要进报告，并作为结构化证据
附加到该事件代表市场的 Agent prompt（LLM 拿到完整分布而非单点）。
"""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, List, Optional, Tuple

from ..models import Market
from ..util import meeting_hint

logger = logging.getLogger(__name__)

_ABOVE = re.compile(
    r"\b(?:above|over|at least|higher than|greater than)\b\s*\$?([\d][\d,]*\.?\d*)", re.IGNORECASE)
_BELOW = re.compile(
    r"\b(?:below|under|less than|lower than)\b\s*\$?([\d][\d,]*\.?\d*)", re.IGNORECASE)
_HIT = re.compile(
    r"hit\s*\(?\s*(high|low)\s*\)?\s*\$?([\d][\d,]*\.?\d*)", re.IGNORECASE)


def _extract(m: Market) -> Optional[Tuple[str, float, str]]:
    """从标题抽 (方向, 阈值, 单位)；非阈值合约返回 None。"""
    q = m.question
    mm = _HIT.search(q)
    if mm:
        direction = "above" if mm.group(1).lower() == "high" else "below"
        return direction, float(mm.group(2).replace(",", "")), ("$" if "$" in q else "%")
    mm = _ABOVE.search(q)
    direction = "above"
    if mm is None:
        mm = _BELOW.search(q)
        direction = "below"
    if mm is None:
        return None
    tail = q[mm.end():mm.end() + 1]
    unit = "%" if tail == "%" else ("$" if "$" in mm.group(0) else "")
    return direction, float(mm.group(1).replace(",", "")), unit


def _label(question: str, direction: str) -> str:
    """系列名 = 标题在第一个数字前的公共前缀 + 方向/会议尾巴。

    同事件不同方向（WTI HIGH/LOW）与同前缀不同会议的系列，只留前缀
    会在报告里出现多行完全相同的 label，无法区分。
    """
    mt = re.search(r"(?<!\w)[\d][\d,.]*%?(?!\w)", question)
    base = question[:mt.start()].strip().rstrip("?:=- ") if mt else question
    base = base[:64]
    if "(high)" not in base.lower() and "(low)" not in base.lower():
        base += "（≥）" if direction == "above" else "（≤）"
    hint = meeting_hint(question)[:40]
    if hint:
        base += f" [{hint}]"
    return base


def median_info(ladder, direction):
    """不外推；不单调/重复阈值不产生中位数，边界使用不等式。"""
    xs = [x for x, _ in ladder]
    ps = [p for _, p in ladder]
    if (not xs or any(not math.isfinite(x) for x in xs)
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in ps)
            or any(b <= a for a, b in zip(xs, xs[1:]))
            or any((b > a if direction == "above" else b < a) for a, b in zip(ps, ps[1:]))):
        return {"median": None, "median_status": "invalid", "median_bound": None}
    for x, p in ladder:
        if p == .5:
            return {"median": x, "median_status": "observed", "median_bound": None}
    for (x, p), (y, q) in zip(ladder, ladder[1:]):
        if (p - .5) * (q - .5) < 0:
            return {"median": round(x + (.5 - p) / (q - p) * (y - x), 3),
                    "median_status": "interpolated", "median_bound": None}
    above_range = (ps[-1] > .5 if direction == "above" else ps[-1] < .5)
    return {"median": None, "median_status": "above_range" if above_range else "below_range",
            "median_bound": xs[-1] if above_range else xs[0]}


def _median(xs, ps, direction):
    return median_info(list(zip(xs, ps)), direction)["median"]


def format_line(median, ladder, direction, unit, comparator=None):
    info = median_info(ladder, direction)
    sym = comparator or (">" if direction == "above" else "<")
    def fmt(x):
        return f"{'$' if unit == '$' else ''}{x:g}{'%' if unit == '%' else ''}"
    status = info["median_status"]
    if status == "invalid":
        summary = "中位数不可估计（阶梯不单调、阈值重复或数据无效）"
    elif info["median"] is not None:
        summary = f"中位数约 {fmt(info['median'])}（离散报价线性估计）"
    else:
        relation = "高于" if status == "above_range" else "低于"
        summary = f"中位数{relation} {fmt(info['median_bound'])}（超出观测范围）"
    lo_x, lo_p = ladder[0]
    hi_x, hi_p = ladder[-1]
    return (f"隐含分布: {summary}；"
            f"P({sym}{fmt(lo_x)})={lo_p*100:.1f}%, P({sym}{fmt(hi_x)})={hi_p*100:.1f}% "
            f"（{len(ladder)} 个阈值合约）")


def summarize(markets: List[Market], min_members: int = 3) -> List[Dict]:
    groups = {}
    for m in markets:
        tv = _extract(m)
        if tv is None:
            continue
        # 联合条件的概率不是单一变量的 CDF。
        if re.search(r"\b(and|or|if|unless)\b", m.question, re.I):
            continue
        direction, value, unit = tv
        inclusive = bool(_HIT.search(m.question) or re.search(r"\bat least\b", m.question, re.I))
        comparator = ("≥" if inclusive else ">") if direction == "above" else ("≤" if inclusive else "<")
        scope = ("期间最高值" if direction == "above" else "期间最低值") if _HIT.search(m.question) else "结算变量"
        key = (m.platform, m.event_key or m.market_id, direction, unit, comparator, scope)
        groups.setdefault(key, []).append((value, m.yes_probability, m))
    out = []
    for (platform, ekey, direction, unit, comparator, scope), pts in groups.items():
        if len(pts) < min_members:
            continue
        pts.sort(key=lambda t: t[0])
        ladder = [(x, p) for x, p, _ in pts]
        info = median_info(ladder, direction)
        out.append({"platform": platform, "event_key": ekey,
                    "label": f"{pts[0][2].question} [{scope}]", "direction": direction,
                    "unit": unit, "comparator": comparator, "scope": scope, "n": len(pts),
                    **info, "ladder": ladder,
                    "total_volume": sum(m.volume_total for _, _, m in pts),
                    "line": f"{scope}：" + format_line(info["median"], ladder, direction, unit, comparator)})
    out.sort(key=lambda s: -s["total_volume"])
    return out
