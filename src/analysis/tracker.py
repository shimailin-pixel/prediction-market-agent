"""概率 / 成交量变化追踪。

异动检测依赖「历史 vs 当前」的对比。这里用 SnapshotStore 的历史快照实现两类动态异动：
- prob_spike：当前概率相对上一快照跳变 ≥ 阈值（百分点）；
- volume_spike：按区间秒数归一化的成交速率放大；Kalshi 使用合约量并要求区间换手率达标。

首次运行无历史时返回空（属正常，随运行逐步积累）。
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

from ..models import Anomaly, Market

logger = logging.getLogger(__name__)


def track_changes(markets: List[Market], store, cfg: Dict[str, Any], now_ts: int) -> List[Anomaly]:
    """对比历史快照，产出 prob_spike / volume_spike 异动。"""
    sig_cfg = cfg.get("signals", {})
    prob_thresh = float(sig_cfg.get("prob_spike_pts", 15.0))
    vol_multiple = float(sig_cfg.get("volume_multiple", 3.0))

    turnover_min = float(sig_cfg.get("kalshi_min_interval_turnover", 0.10))
    if not math.isfinite(turnover_min) or turnover_min < 0:
        raise ValueError("kalshi_min_interval_turnover must be finite and >= 0")

    anomalies: List[Anomaly] = []
    for m in markets:
        hist = store.history(m.market_id)
        # 历史里剔除本次即将写入的当前点：取 before now_ts
        prior = [h for h in hist if h["ts"] < now_ts]
        if not prior:
            continue
        prev = prior[-1]

        # ---- 概率跳变 ----
        if prev.get("yes_probability") is not None:
            delta_pts = (m.yes_probability - prev["yes_probability"]) * 100
            if abs(delta_pts) >= prob_thresh:
                anomalies.append(Anomaly(
                    market_id=m.market_id, platform=m.platform, question=m.question,
                    anomaly_type="prob_spike", severity="high",
                    current_probability=m.yes_probability,
                    details={
                        "old_probability": round(prev["yes_probability"], 4),
                        "new_probability": round(m.yes_probability, 4),
                        "change_pts": round(delta_pts, 1),
                        "direction": "up" if delta_pts > 0 else "down",
                        "minutes_since_prev": round((now_ts - prev["ts"]) / 60, 1),
                    },
                    market=m,
                ))

        # Kalshi 仅使用原始合约数；旧快照缺失时等待新基线，禁止回退美元估计。
        field = "volume_contracts" if m.platform == "kalshi" else "volume_total"
        current = getattr(m, field)
        def valid(value):
            return value is not None and math.isfinite(value) and value >= 0

        if len(prior) < 3 or not valid(current) or not valid(prev.get(field)):
            continue
        elapsed = now_ts - prev["ts"]
        increment = current - prev[field]
        if elapsed <= 0 or increment <= 0:
            continue
        # 按时间归一化，防止漏跑或运行间隔变长被误判成放量。
        rates = []
        for left, right in zip(prior, prior[1:]):
            dt = right["ts"] - left["ts"]
            if not valid(left.get(field)) or not valid(right.get(field)) or dt <= 0:
                rates = []  # 缺失/重置之前的历史不混进新基线
                continue
            delta = right[field] - left[field]
            if delta < 0:
                rates = []
                continue
            rates.append(delta / dt)
        if len(rates) < 2:
            continue
        mean_rate = sum(rates) / len(rates)
        expected = mean_rate * elapsed
        if expected <= 0 or increment < expected * vol_multiple:
            continue
        details = {"multiple": round(increment / expected, 1),
                   "interval_seconds": elapsed}
        if m.platform == "kalshi":
            oi = m.open_interest_contracts
            # 无持仓或缺失持仓无法计算换手率，即使关闭门槛也不伪造数值。
            if not valid(oi) or oi <= 0:
                continue
            turnover = increment / oi
            if turnover < turnover_min:
                continue
            details.update(increment_contracts=round(increment, 4),
                           expected_increment_contracts=round(expected, 4),
                           open_interest_contracts=oi,
                           interval_turnover=turnover,
                           min_interval_turnover=turnover_min,
                           volume_unit="contracts")
        else:
            details.update(increment_usd=round(increment, 2),
                           mean_increment_usd=round(expected, 2), volume_unit="usd")
        anomalies.append(Anomaly(
            market_id=m.market_id, platform=m.platform, question=m.question,
            anomaly_type="volume_spike", severity="medium",
            current_probability=m.yes_probability, details=details, market=m,
        ))

    logger.info("动态追踪: %d 个市场, 检出 %d 个 prob/volume 异动", len(markets), len(anomalies))
    return anomalies
