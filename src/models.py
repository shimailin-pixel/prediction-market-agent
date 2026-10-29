"""跨平台统一数据模型。

两个平台（Polymarket / Kalshi）的原始字段差异极大，但下游分析只关心同一套语义：
问题、YES 概率、成交额、流动性、到期时间、事件类型。这里定义归一化后的 `Market`
作为整个框架的通用货币（single source of truth），平台差异在 data_source 层消化。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Market:
    """归一化后的预测市场。金额一律为 USD，概率一律为 0~1 小数。"""

    market_id: str
    platform: str                      # "polymarket" | "kalshi"
    question: str
    yes_probability: float             # 0~1
    no_probability: float              # 0~1

    volume_24h: float = 0.0            # 近 24h 成交额（USD）
    volume_total: float = 0.0          # 累计成交额（USD）
    liquidity: float = 0.0             # 挂单流动性（USD）；Kalshi 用未平仓合约美元代理

    # ---- 分类（analysis.classify 填充）----
    event_types: List[str] = field(default_factory=list)
    primary_type: str = ""
    question_shape: str = "unknown"    # threshold | event | unknown

    # ---- 时间 ----
    end_date: str = ""
    days_to_resolution: Optional[int] = None

    # ---- 平台专属句柄（用于增强 / 去重 / 跨平台）----
    event_key: str = ""                # 事件归组键：PM event slug / Kalshi event_ticker
    condition_id: str = ""             # Polymarket
    token_id: str = ""                 # Polymarket CLOB YES token
    ticker: str = ""                   # Kalshi
    category: str = ""                 # Kalshi event category
    slug: str = ""                     # Polymarket event slug
    url: str = ""
    description: str = ""

    # ---- 增强 / 追踪字段（运行中填充）----
    volume_24h_share: Optional[float] = None    # 24h 成交 / 累计成交
    prob_change_24h_pts: Optional[float] = None  # 24h 概率变动（百分点）
    prob_change_7d_pts: Optional[float] = None   # 7d 概率变动（百分点）
    concentration: Optional[Dict[str, Any]] = None  # 持仓集中度（Polymarket）
    snapshot_prob: Optional[float] = None       # 上一快照概率（用于相邻跳变）
    volume_contracts: Optional[float] = None   # Kalshi 累计成交合约数，禁止用当前价格重估后求差
    open_interest_contracts: Optional[float] = None  # Kalshi 未平仓合约数

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def yes_pct(self) -> float:
        return round(self.yes_probability * 100, 1)


@dataclass
class Anomaly:
    """检测出的异动 / 信号，供 Agent 判断层与输出层消费。"""

    market_id: str
    platform: str
    question: str
    anomaly_type: str                  # volume_spike | prob_spike | prob_momentum | money_inflow | whale | cross_platform_divergence
    severity: str                      # high | medium
    current_probability: float         # 0~1
    details: Dict[str, Any] = field(default_factory=dict)
    market: Optional[Market] = None    # 附带完整市场上下文

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["market"] = self.market.to_dict() if self.market else None
        return d
