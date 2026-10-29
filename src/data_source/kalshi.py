"""Kalshi 数据接入。

关键工程决策（实测验证，2026 Q1 API 变更后）：
1. 端点：生产域名 external-api.kalshi.com/trade-api/v2，公开 /markets 无需认证。
2. 取数：GET /markets + 成交时间窗口 min_close_ts / max_close_ts。这一招同时解决三个坑——
   - 默认 /markets 列表被 KXMVECROSSCATEGORY 串关组合污染 → 时间窗口天然排掉它们；
   - /events 默认捞到的多是「登火星 2099」类超长周期噱头盘（无日成交量）；
   - 窗口保证到期日在范围内，直接通过「未过期」筛选。
3. **query 参数被 API 静默忽略**（实测 query=Iran 与 query=Fed 返回完全相同），且默认
   按 close_time **从远到近**排序：长窗口下前几十页全是 378+ 天的远期选举/体育盘，
   近端 Fed/宏观盘永远翻不到。故按「近细远粗」连续分段扫描，筛选交给本地相关性门禁。
4. 字段迁移：整数字段（yes_bid / volume / liquidity）已移除，改为字符串定点
   （yes_bid_dollars / volume_fp / volume_24h_fp / open_interest_fp），需安全解析。
5. 单位：1 张合约到期支付 $1，故美元口径 ≈ 合约数 × YES 概率。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from ..http import HttpClient, HttpError
from ..models import Market
from ..util import safe_float, days_until

logger = logging.getLogger(__name__)


def _is_parlay(m: dict) -> bool:
    et = m.get("event_ticker") or ""
    return bool(m.get("mve_selected_legs")) or et.startswith("KXMVECROSSCATEGORY")


class KalshiClient:
    def __init__(self, cfg: Dict[str, Any], http: Optional[HttpClient] = None):
        base = cfg.get("data_sources", {}).get("kalshi", {})
        self.base = base.get("base", "https://external-api.kalshi.com/trade-api/v2").rstrip("/")
        self.max_close_days = int(base.get("max_close_days", 400))
        self.per_page = int(base.get("per_page", 500))
        # 分段扫描参数：近端窗口切细（宏观盘集中在 60 天内），远端窗口粗，控制请求量
        self.near_segment_days = int(base.get("near_segment_days", 7))
        self.far_segment_days = int(base.get("far_segment_days", 30))
        self.near_horizon_days = int(base.get("near_horizon_days", 60))
        self.max_pages_per_segment = int(base.get("max_pages_per_segment", 4))
        self.http = http or HttpClient()

    def _yes_probability(self, m: dict) -> Optional[float]:
        """YES 概率（0~1）：优先 mid，回落 last_price。"""
        bid = safe_float(m.get("yes_bid_dollars"), 0.0)
        ask = safe_float(m.get("yes_ask_dollars"), 0.0)
        if 0 < bid <= ask and ask <= 1:
            return (bid + ask) / 2
        last = safe_float(m.get("last_price_dollars"), -1.0)
        if 0 <= last <= 1:
            return last
        return None

    def _usd_volume(self, contracts_fp: float, prob: float) -> float:
        """合约数（定点）× 概率 ≈ 美元成交额估计。"""
        return safe_float(contracts_fp, 0.0) * prob

    def _segments(self, now: int):
        """近细远粗的连续成交时间窗口：60 天内每 near_segment_days 一段，之外每 far_segment_days 一段。"""
        segs = []
        end = now + max(1, self.max_close_days) * 86400
        near_until = now + self.near_horizon_days * 86400
        t = now
        while t < end:
            days = self.near_segment_days if t < near_until else self.far_segment_days
            seg_end = min(t + max(1, days) * 86400, end)
            segs.append((t, seg_end))
            t = seg_end
        return segs

    def _to_market(self, m: dict, seen: set) -> Optional[Market]:
        """原始 dict → 归一化 Market；串关/重复/无价格返回 None。"""
        if _is_parlay(m):
            return None
        prob = self._yes_probability(m)
        if prob is None:
            return None
        ticker = m.get("ticker", "")
        if not ticker or ticker in seen:
            return None
        seen.add(ticker)

        vol_total = self._usd_volume(m.get("volume_fp"), prob)
        vol_24h = self._usd_volume(m.get("volume_24h_fp"), prob)
        open_int = self._usd_volume(m.get("open_interest_fp"), prob)
        close_ts = m.get("close_time") or m.get("expiration_time") or ""
        return Market(
            market_id=ticker,
            platform="kalshi",
            question=m.get("title", "") or m.get("subtitle", ""),
            yes_probability=prob,
            no_probability=round(1 - prob, 4),
            volume_24h=vol_24h,
            volume_total=vol_total,
            volume_contracts=safe_float(m.get("volume_fp"), None),
            open_interest_contracts=safe_float(m.get("open_interest_fp"), None),
            liquidity=open_int,  # 未平仓合约美元作为流动性代理
            end_date=close_ts,
            days_to_resolution=days_until(close_ts),
            ticker=ticker,
            event_key=m.get("event_ticker") or m.get("series_ticker") or ticker,
            category=m.get("category", "") or "",
            description=(m.get("subtitle", "") or "")[:400],
            url=f"https://kalshi.com/markets/{(m.get('event_ticker') or ticker).lower()}",
        )

    def fetch_markets(self, keywords: List[str], cfg: Dict[str, Any]) -> List[Market]:
        """分段 + cursor 翻页扫描成交时间窗口，归一化为 Market。

        注意：/markets 的 query 参数实测被忽略，服务端关键词检索不可用；
        关键词的精筛由本地门禁（signals.filter_relevant）完成，这里只负责拉全窗口。
        """
        markets: List[Market] = []
        seen: set = set()
        now = int(time.time())
        limit = min(max(self.per_page, 200), 1000)
        segs = self._segments(now)
        pages_used = 0

        for seg_start, seg_end in segs:
            cursor: Optional[str] = None
            for _page in range(max(1, self.max_pages_per_segment)):
                params: Dict[str, Any] = {
                    "status": "open",
                    "limit": limit,
                    "min_close_ts": seg_start,
                    "max_close_ts": seg_end,
                }
                if cursor:
                    params["cursor"] = cursor
                try:
                    data = self.http.get_json(f"{self.base}/markets", params)
                except HttpError as e:
                    logger.error("[kalshi] /markets 失败 (%s~%s): %s", seg_start, seg_end, e)
                    break
                pages_used += 1
                raw = data.get("markets", []) if isinstance(data, dict) else []
                for m in raw:
                    km = self._to_market(m, seen)
                    if km is not None:
                        markets.append(km)
                cursor = data.get("cursor") if isinstance(data, dict) else None
                if not raw or not cursor:
                    break

        logger.info("[kalshi] 分段扫描 %d 段/%d 页，新增 %d 个市场（query 参数无效，筛选交给本地门禁）",
                    len(segs), pages_used, len(markets))
        return markets

    def close(self) -> None:
        self.http.close()
