"""信号层：有价值市场过滤 + 聪明钱信号 + 资产配置建议。

- filter_valuable：剔除已过期 / 零流动性 / 长尾死盘。这一步把之前监控里
  「63% 是已结算历史盘、信号被 0%/100% 噪音淹没」的问题从源头收住；
- detect_smart_money：基于当前快照的静态聪明钱信号（放量、资金涌入、概率倾斜、巨鲸集中）；
  概率动量优先用增强层拿到的 24h/7d 变化；
- generate_asset_suggestions：按 primary_type 聚合，区分 threshold/event，取高信念 Top-N。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from ..models import Anomaly, Market
from ..util import format_volume
from .classify import EVENT_ASSET_MAP, classify_signal_strength, type_from_keyword
from .elections import is_off_theme

logger = logging.getLogger(__name__)

# 利率动作必须区分方向；维持不变/无加息并不等价于降息。
_RATE_CUT = re.compile(r"\b(cut|cuts|decrease[sd]?|lower(?:s|ed)?)\b", re.IGNORECASE)
_RATE_HIKE = re.compile(r"\b(hike[sd]?|increase[sd]?|raise[sd]?)\b", re.IGNORECASE)
_RATE_NEGATION = re.compile(r"\b(no|not|never|neither|without)\b|n['’]t\b", re.IGNORECASE)
_RATE_HOLD = re.compile(r"\b(hold|holds|pause|pauses|unchanged|stand pat|no change)\b", re.IGNORECASE)

# 「低概率=风险消退」只是现状推论而非缓和事件本身的地缘类型：
# 建议方向需人工复核（如 invasion 盘 1.8% 不等于可以买入风险资产）。
_STATUS_QUO_LOW_TYPES = {"war", "iran", "russia", "china", "mideast"}


def _rate_bucket(question: str, prob: float, high: float, low: float,
                 tail_floor: float, stale_low: float, stale_high: float) -> Optional[str]:
    """将明确的加息/降息命题映射到鹰派强度，含糊命题不出方向建议。"""
    cut = bool(_RATE_CUT.search(question))
    hike = bool(_RATE_HIKE.search(question))
    if (_RATE_NEGATION.search(question) or _RATE_HOLD.search(question)
            or cut == hike or re.search(r"\b0+(?:\.0+)?\s*(?:bp|bps|basis points?|%)", question, re.I)):
        return None
    # 低概率降息的补集含维持不变，不能映射为加息；反之亦然。
    if prob < high:
        return None
    return "low" if cut else "high"


def suggestion_block_reason(m: Market) -> str:
    """保守建议门禁；保持市场可监测，仅阻止从含糊命题生成交易方向。"""
    q = m.question
    if re.search(r"\b(and|or|either|both|if|unless)\b|&", q, re.I):
        return "联合或条件命题，不能拆成单一事件概率"
    if m.question_shape == "threshold":
        return "价格/利率水平或触及阈值不等于未来涨跌方向"
    if m.primary_type in {"iran", "china", "russia", "mideast", "oil", "crypto", "equity_index", "nuclear_agreement"}:
        return "主题分类不足以确定资产方向"
    if _RATE_NEGATION.search(q):
        return "否定命题的补集不直接映射成相反方向"
    if m.primary_type == "interest_rate":
        if _rate_bucket(q, m.yes_pct, 60, 30, 2, 40, 60) is None:
            return "缺乏明确且高概率的非零加息/降息事件"
    elif m.yes_probability < .60:
        return "低概率或中性定价不能证明相反事件发生"
    if m.primary_type == "nuclear" and not re.search(r"\b(weapon|weapons|warhead|attack|bomb|icbm)\b", q, re.I):
        return "核主题未明确指向武器/攻击风险"
    return ""


def anomaly_priority(a):
    """真实概率变化、真实成交速率变化优先于静态市场规模。"""
    tags = set(a.anomaly_type.split("+"))
    rank = (0 if tags & {"prob_spike", "prob_momentum", "prob_trend"}
            else 1 if "volume_spike" in tags else 2 if "cross_platform_divergence" in tags else 3)
    return (rank, a.severity != "high", -abs(a.details.get("change_pts", 0)),
            -float(a.details.get("volume_total") or (a.market.volume_total if a.market else 0)))


def filter_valuable(markets: List[Market], cfg: Dict[str, Any]) -> List[Market]:
    """保留：未过期 + 有流动性 + 累计成交额达标。

    阈值支持按平台覆盖（runtime.per_source.<platform>.min_volume/min_liquidity）：
    Kalshi 的美元成交额是「合约数×概率」估计值，量级天然小于 Polymarket。
    """
    rt = cfg.get("runtime", {})
    min_volume = float(rt.get("min_volume", 0))
    min_liquidity = float(rt.get("min_liquidity", 1))
    per_source = rt.get("per_source", {}) or {}

    kept: List[Market] = []
    for m in markets:
        ov = per_source.get(m.platform, {}) or {}
        mv = float(ov.get("min_volume", min_volume))
        ml = float(ov.get("min_liquidity", min_liquidity))
        if m.days_to_resolution is not None and m.days_to_resolution < 0:
            continue  # 已过期 / 已结算
        if m.liquidity < ml:
            continue  # 零流动性死盘
        if m.volume_total < mv:
            continue  # 长尾
        kept.append(m)

    logger.info(
        "有价值过滤: 原始=%d, 保留=%d（剔除已过期/零流动性/长尾）",
        len(markets), len(kept),
    )
    return kept


def filter_relevant(markets: List[Market], cfg: Dict[str, Any]) -> List[Market]:
    """关键词相关性过滤：只保留与政治/经济/财经主题直接相关的市场。

    平台侧搜索（尤其 Kalshi 的 query 参数）很宽松，搜 'Iran' 能回 500 个盘；
    这里在本地做精确门禁：
    - 体育/娱乐/奖项类无条件剔除（is_off_theme）；
    - 有关键词时：事件类型命中「关键词→类型」映射，或原文（问题/描述/分类）
      词边界命中关键词，二者居一才保留；
    - 关键词为空时（空配置兜底）：只保留命中分类体系（primary_type 非空）的市场，
      即仍限定在地缘/宏观/金融主题内，不放进棒球赛程这种东西。
    """
    rt = cfg.get("runtime", {})
    keywords = [str(k).strip() for k in (rt.get("keywords") or []) if str(k).strip()]
    strict = bool(rt.get("require_keyword_match", True))

    allowed_types = {t for t in (type_from_keyword(k) for k in keywords) if t}
    kw_pats = [re.compile(r"\b" + re.escape(k.lower()) + r"\b") for k in keywords]

    kept: List[Market] = []
    n_off, n_irrelevant = 0, 0
    for m in markets:
        if is_off_theme(m.question):
            n_off += 1
            continue
        if not strict:
            kept.append(m)
            continue
        if not keywords:
            matched = bool(m.primary_type)
        else:
            text = " ".join(filter(None, [m.question, m.description, m.category, m.slug])).lower()
            matched = bool(set(m.event_types) & allowed_types) or any(p.search(text) for p in kw_pats)
        if matched:
            kept.append(m)
        else:
            n_irrelevant += 1

    logger.info(
        "关键词相关性过滤: 原始=%d, 保留=%d（剔除体育娱乐奖项 %d、无关 %d；关键词 %d 个）",
        len(markets), len(kept), n_off, n_irrelevant, len(keywords),
    )
    return kept


def quota_truncate(
    markets: List[Market], limit: int, min_per_platform: int
) -> List[Market]:
    """按成交额截断，但为每个平台保底至少 min_per_platform 个。

    防止量级差异（Kalshi 成交额估计值远小于 Polymarket 真实 USD）
    导致小平台在全局排序中被完全淹没。
    """
    ordered = sorted(markets, key=lambda m: m.volume_total, reverse=True)
    if min_per_platform <= 0:
        return ordered[:limit]
    reserved, seen, count = [], set(), {}
    for m in ordered:
        c = count.get(m.platform, 0)
        if c < min_per_platform:
            reserved.append(m)
            count[m.platform] = c + 1
            seen.add(id(m))
    rest = [m for m in ordered if id(m) not in seen]
    out = reserved + rest[: max(0, limit - len(reserved))]
    return sorted(out, key=lambda m: m.volume_total, reverse=True)


def dedupe_anomalies_by_event(anomalies: List[Anomaly], per_event_max: int = 1) -> List[Anomaly]:
    """事件级去重：同一事件（同平台 + event_key）的多个合约只保留 Top-N 异动。

    Fed 利率阶梯这类互斥合约是一个概率分布的切片，全部进 Agent 会霸占槽位
    （同一事件解读 8 遍并不会多信息）。event_key 为空的不参与聚合。
    组内代表选择：severity 优先，同级取累计成交额更大者。
    """
    if per_event_max <= 0:
        return list(anomalies)
    sev = {"high": 0, "medium": 1, "low": 2}
    ranked = sorted(
        anomalies,
        key=anomaly_priority,
    )
    kept: List[Anomaly] = []
    count: Dict[tuple, int] = {}
    for a in ranked:
        key = (a.platform, a.market.event_key) if (a.market and a.market.event_key) else None
        if key is None:
            kept.append(a)
            continue
        c = count.get(key, 0)
        if c < per_event_max:
            count[key] = c + 1
            kept.append(a)
    dropped = len(anomalies) - len(kept)
    if dropped:
        logger.info("事件级去重: %d → %d 条（同事件互斥合约只留代表，per_event_max=%d）",
                    len(anomalies), len(kept), per_event_max)
    return kept


def detect_smart_money(markets: List[Market], cfg: Dict[str, Any]) -> List[Anomaly]:
    """静态聪明钱信号（不依赖历史快照；历史类异动见 tracker.track_changes）。"""
    if not markets:
        return []
    sc = cfg.get("signals", {})
    vol_multiple = float(sc.get("volume_multiple", 3.0))
    inflow_share = float(sc.get("volume_24h_share", 0.08))
    mom_24h = float(sc.get("prob_change_24h_pts", 5.0))
    mom_7d = float(sc.get("prob_change_7d_pts", 10.0))
    conc_min = float(sc.get("concentration_top1_pct", 45.0))

    volumes = [m.volume_total for m in markets if m.volume_total > 0]
    avg_volume = sum(volumes) / len(volumes) if volumes else 0.0
    # 放量基准按平台各自计算：避免被 Polymarket 的真实 USD 量级拉高后，
    # Kalshi（成交额为合约数×概率估计值）永远达不到 3× 均值
    plat_vols: Dict[str, List[float]] = {}
    for m in markets:
        if m.volume_total > 0:
            plat_vols.setdefault(m.platform, []).append(m.volume_total)
    avg_by_plat = {p: sum(v) / len(v) for p, v in plat_vols.items()}

    anomalies: List[Anomaly] = []
    for m in markets:
        reasons: List[str] = []
        tags: List[str] = []
        vol_avg = avg_by_plat.get(m.platform, avg_volume)

        # 24h 成交占比（资金正在涌入）
        if m.volume_total > 0 and m.volume_24h > 0:
            share = m.volume_24h / m.volume_total
            m.volume_24h_share = round(share, 4)
            if share >= inflow_share:
                reasons.append(f"24h 成交占累计 {share*100:.1f}%（{format_volume(m.volume_24h)}），成交占比较高（不代表净资金流入）")
                tags.append("money_inflow")

        # 累计放量（对比本平台自己的平均成交额）
        if vol_avg and m.volume_total > vol_avg * vol_multiple:
            reasons.append(f"累计成交 {format_volume(m.volume_total)} 远超本平台均值 {format_volume(vol_avg)}（{m.volume_total / vol_avg:.1f}×）")
            tags.append("volume")

        # 概率动量（来自增强层 CLOB 历史）
        if m.prob_change_24h_pts is not None and abs(m.prob_change_24h_pts) >= mom_24h:
            reasons.append(f"24h 概率{'上调' if m.prob_change_24h_pts>0 else '下调'} {abs(m.prob_change_24h_pts):.1f} 点，正在重定价")
            tags.append("prob_momentum")
        elif m.prob_change_7d_pts is not None and abs(m.prob_change_7d_pts) >= mom_7d:
            reasons.append(f"7d 概率{'上调' if m.prob_change_7d_pts>0 else '下调'} {abs(m.prob_change_7d_pts):.1f} 点，中期趋势明确")
            tags.append("prob_trend")

        # 巨鲸集中
        if m.concentration and m.concentration.get("top1_pct", 0) >= conc_min:
            reasons.append(f"最大单一钱包占可见头部 {m.concentration['top1_pct']:.1f}%，仅为可见样本集中度，不证明净买入或方向")
            tags.append("whale")

        if not tags:
            continue

        anomalies.append(Anomaly(
            market_id=m.market_id, platform=m.platform, question=m.question,
            anomaly_type="+".join(tags),
            severity="high" if len(tags) >= 2 else "medium",
            current_probability=m.yes_probability,
            details={
                "signal_reasons": reasons,
                "signal_tags": tags,
                "volume_total": m.volume_total,
                "volume_24h": m.volume_24h,
                "liquidity": m.liquidity,
                "signal_strength": classify_signal_strength(m.yes_pct),
                "primary_type": m.primary_type,
                "question_shape": m.question_shape,
                "days_to_resolution": m.days_to_resolution,
                "prob_change_24h_pts": m.prob_change_24h_pts,
                "prob_change_7d_pts": m.prob_change_7d_pts,
            },
            market=m,
        ))

    anomalies.sort(key=lambda a: (a.severity != "high", -len(a.details.get("signal_tags", []))))
    logger.info("聪明钱信号: %d 条", len(anomalies))
    return anomalies


def observation_action(event_type, bucket):
    """将规则映射表达为待确认的传导观察，不把概率直接写成交易指令。"""
    actions = {
        "war": "观察黄金、能源及风险资产的风险溢价，等待价格确认",
        "ceasefire": "观察能源和避险资产的风险溢价是否回落",
        "nuclear": "观察尾部风险定价及避险资产反应",
        "tariff": "观察出口链成本、订单与资产价格反应",
        "sanctions": "核对制裁范围及相关产业链暴露",
        "recession": "观察增长预期、信用利差和周期资产反应",
        "inflation": "观察通胀数据、实际利率及成本传导",
        "interest_rate": ("观察宽松预期及成长资产反应" if bucket == "low"
                          else "观察美债收益率、美元及高久期资产反应"),
    }
    return actions.get(event_type, "等待事件和资产价格共同确认")


def generate_asset_suggestions(markets: List[Market], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按事件类型聚合，输出 Top-N 资产配置建议。

    三道闸防噪：
    1. 事件聚合：同 event_key 的互斥合约只用代表市场（volume 最大）参与建议；
    2. 尾概率下限：low 桶不再接受 < tail_floor% 的尾部彩票合约（它们不是
       「弱预期」而是「市场认为不可能」，无交易含义）；
    3. 24h 活跃门槛：累计有量但近期无成交的噱头/死盘不配出建议。
    """
    rt = cfg.get("runtime", {})
    top_n_per_type = int(rt.get("suggest_top_n", 3))
    min_24h = float(rt.get("suggest_min_volume_24h", 500))
    tail_floor = float(rt.get("suggest_tail_prob_floor", 2.0))
    high, low = 60.0, 30.0
    stale_low, stale_high = 40.0, 60.0

    # 1. 事件聚合：每 (platform, event_key) 取累计成交额最大的合约作代表
    best_by_event: Dict[tuple, Market] = {}
    for m in markets:
        if suggestion_block_reason(m):
            continue
        key = (m.platform, m.event_key or m.market_id)
        cur = best_by_event.get(key)
        if cur is None or m.volume_total > cur.volume_total:
            best_by_event[key] = m

    buckets: Dict[str, List[Market]] = {}
    for m in best_by_event.values():
        if not m.primary_type:
            continue
        if m.volume_24h < min_24h:  # 3. 活跃门槛
            continue
        buckets.setdefault(m.primary_type, []).append(m)

    suggestions: List[Dict[str, Any]] = []
    for tname, cands in buckets.items():
        rules = EVENT_ASSET_MAP.get(tname)
        if not rules:
            continue
        scored = []
        for m in cands:
            prob = m.yes_pct
            if m.primary_type == "interest_rate":
                bucket = _rate_bucket(m.question, prob, high, low, tail_floor, stale_low, stale_high)
                if bucket is None:
                    continue
            else:
                if prob >= high:
                    bucket = "high"
                elif tail_floor <= prob <= low:
                    bucket = "low"   # 2. 尾部彩票（prob < tail_floor）不再作为信号
                elif stale_low <= prob < stale_high:
                    bucket = "mid"
                else:
                    continue
            action = rules.get(bucket)
            if not action:
                continue
            scored.append((abs(prob - 50.0), m, bucket, action))
        if not scored:
            continue
        scored.sort(key=lambda x: (-x[0], -x[1].volume_total))
        for conviction, m, bucket, action in scored[:top_n_per_type]:
            needs_review = (
                m.question_shape == "threshold"
                or (tname == "interest_rate" and bool(_RATE_NEGATION.search(m.question)))
                or (tname in _STATUS_QUO_LOW_TYPES and bucket == "low")
            )
            suggestions.append({
                "event_type": tname,
                "direction": bucket,
                "action": observation_action(tname, bucket),
                "reason": "条件性传导假设，尚未核对资产现价与基准预期",
                "trigger": f"{m.question} (Yes {m.yes_pct}%)",
                "platform": m.platform,
                "probability": m.yes_pct,
                "volume": format_volume(m.volume_total),
                "conviction": round(conviction, 1),
                "days_to_resolution": m.days_to_resolution,
                "needs_review": True,
                "basis": "条件性规则映射，未估计超额收益",
            })

    order = {"high": 0, "mid": 1, "low": 2}
    suggestions.sort(key=lambda s: (order.get(s["direction"], 3), -s["conviction"]))
    return suggestions
