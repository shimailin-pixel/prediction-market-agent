"""Polymarket 数据接入。

三个公开 API（均无需认证）：
- Gamma  (元数据)：/public-search 关键词检索 → events → markets（问题、概率、成交额、流动性）
- CLOB   (概率历史)：/prices-history → YES token 概率时间序列（识别概率重定价）
- Data   (持仓)：/holders 或 /v2/holders → 头部持仓集中度（识别巨鲸）

解析要点（沿用 polymarket_monitor 的健壮处理）：
- 按 outcomes 标签定位 Yes/No，不硬取 prices[0]/[1]；
- 非二元（多结果）市场拒绝解析，避免把选举类算成 Yes/No 概率；
- 缺失 outcomePrices 时用 bestBid/bestAsk 兜底。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from ..http import HttpClient, HttpError
from ..models import Market
from ..util import safe_float, parse_json_field, days_until

logger = logging.getLogger(__name__)


class PolymarketClient:
    def __init__(self, cfg: Dict[str, Any], http: Optional[HttpClient] = None):
        base = cfg.get("data_sources", {}).get("polymarket", {})
        self.gamma = base.get("gamma_base", "https://gamma-api.polymarket.com")
        self.clob = base.get("clob_base", "https://clob.polymarket.com")
        self.data = base.get("data_base", "https://data-api.polymarket.com")
        self.http = http or HttpClient()
        self.throttle = 0.25
        self._last = 0.0

    def _pace(self) -> None:
        gap = time.time() - self._last
        if gap < self.throttle:
            time.sleep(self.throttle - gap)
        self._last = time.time()

    # ---- Gamma 检索 ----

    def search_events(self, keyword: str, limit: int) -> List[dict]:
        """优先 /public-search；空则回落 /events?search=。"""
        self._pace()
        try:
            data = self.http.get_json(
                f"{self.gamma}/public-search", {"q": keyword, "limit_per_type": limit}
            )
            events = data.get("events", []) if isinstance(data, dict) else []
            if events:
                return events
        except HttpError:
            pass

        self._pace()
        try:
            data = self.http.get_json(
                f"{self.gamma}/events",
                {"active": "true", "closed": "false", "search": keyword, "limit": limit},
            )
            return data if isinstance(data, list) else []
        except HttpError:
            return []

    # ---- 归一化 ----

    @staticmethod
    def _resolve_yes_no(market: dict) -> Optional[tuple]:
        outcomes = parse_json_field(market.get("outcomes"), None)
        prices = parse_json_field(market.get("outcomePrices"), None)
        if not isinstance(prices, list) or not prices:
            best_ask = safe_float(market.get("bestAsk"), -1.0)
            best_bid = safe_float(market.get("bestBid"), -1.0)
            if best_ask >= 0:
                return best_ask, max(0.0, 1.0 - best_bid)
            return None
        if isinstance(outcomes, list) and outcomes:
            if len(outcomes) != len(prices):
                return None
            labels = [str(o).strip().lower() for o in outcomes]
            yes_idx = next((i for i, o in enumerate(labels) if o == "yes"), None)
            no_idx = next((i for i, o in enumerate(labels) if o == "no"), None)
            if yes_idx is None or no_idx is None:
                return None  # 非二元市场
            return safe_float(prices[yes_idx]), safe_float(prices[no_idx])
        if len(prices) == 2:
            return safe_float(prices[0]), safe_float(prices[1])
        return None

    def fetch_markets(self, keywords: List[str], cfg: Dict[str, Any]) -> List[Market]:
        """按关键词检索并归一化为 Market 列表（分类留给 analysis 层）。"""
        runtime = cfg.get("runtime", {})
        top_n = int(runtime.get("top_n_per_keyword", 25))
        markets: List[Market] = []
        seen: set = set()

        for kw in keywords:
            events = self.search_events(kw, top_n)
            for event in events:
                for m in event.get("markets", []) or []:
                    yn = self._resolve_yes_no(m)
                    if yn is None:
                        continue
                    yes_p, no_p = yn
                    mid = str(m.get("id", ""))
                    if not mid or mid in seen:
                        continue
                    seen.add(mid)

                    clob_ids = parse_json_field(m.get("clobTokenIds"), []) or []
                    markets.append(
                        Market(
                            market_id=mid,
                            platform="polymarket",
                            question=m.get("question", "") or event.get("title", ""),
                            yes_probability=yes_p,
                            no_probability=no_p,
                            volume_24h=safe_float(m.get("volume24hr"), 0.0),
                            volume_total=safe_float(m.get("volumeNum", m.get("volume")), 0.0),
                            liquidity=safe_float(m.get("liquidityNum", m.get("liquidity")), 0.0),
                            end_date=m.get("endDate", "") or m.get("endDateIso", ""),
                            days_to_resolution=days_until(m.get("endDate", "")),
                            condition_id=m.get("conditionId", "") or "",
                            token_id=clob_ids[0] if clob_ids else "",
                            event_key=event.get("slug", "") or "",
                            slug=event.get("slug", "") or "",
                            description=(m.get("description", "") or "")[:400],
                            url=f"https://polymarket.com/event/{event.get('slug','')}",
                        )
                    )
            logger.info("[polymarket] 关键词 '%s' 累计解析 %d 个市场", kw, len(markets))
        return markets

    # ---- 增强：概率历史 ----

    def fetch_prob_history(self, token_id: str, interval: str = "1w", fidelity: int = 60) -> Optional[dict]:
        if not token_id:
            return None
        self._pace()
        try:
            data = self.http.get_json(
                f"{self.clob}/prices-history",
                {"market": token_id, "interval": interval, "fidelity": fidelity},
            )
        except HttpError:
            return None
        history = (data or {}).get("history") or []
        if len(history) < 2:
            return None
        pts = sorted(((int(p.get("t", 0)), safe_float(p.get("p"))) for p in history))
        now_ts = int(time.time())
        last_p = pts[-1][1]
        first_p = pts[0][1]
        day_ago = next((p for ts, p in pts if ts >= now_ts - 86400), first_p)
        return {
            "last": round(last_p, 4),
            "change_24h_pts": round((last_p - day_ago) * 100, 2),
            "change_7d_pts": round((last_p - first_p) * 100, 2),
            "points": len(pts),
        }

    # ---- 增强：持仓集中度 ----

    def fetch_holder_concentration(self, condition_id: str, top_n: int = 5) -> Optional[dict]:
        """最大单一钱包 / 可见 Top-N 之和（仅头部样本，不外推全体）。"""
        if not condition_id:
            return None
        payload = None
        for url, params in (
            (f"{self.data}/v2/holders", {"condition": condition_id, "limit": 50}),
            (f"{self.data}/holders", {"market": condition_id, "limit": 20}),
        ):
            self._pace()
            try:
                payload = self.http.get_json(url, params)
                break
            except HttpError:
                continue
        if payload is None:
            return None

        groups = payload.get("data", []) if isinstance(payload, dict) else payload
        if not isinstance(groups, list) or not groups:
            return None

        best = None
        for group in groups:
            holders = group.get("holders") or []
            amounts = sorted((safe_float(h.get("amount")) for h in holders), reverse=True)
            visible = sum(amounts)
            if visible <= 0:
                continue
            cand = {
                "top1_pct": round(amounts[0] / visible * 100, 2),
                "top3_pct": round(sum(amounts[:3]) / visible * 100, 2),
                "top_n_pct": round(sum(amounts[:top_n]) / visible * 100, 2),
                "holder_count": len(holders),
            }
            if best is None or cand["top1_pct"] > best["top1_pct"]:
                best = cand
        return best

    def close(self) -> None:
        self.http.close()
