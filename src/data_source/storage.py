"""概率 / 成交量历史快照存储（SQLite 标准库，零依赖）。

追踪「概率变化」的前提是有历史。每次运行把当前 Market 的关键指标落一张快照，
下次运行即可对比：
- 相邻快照的概率跳变（prob_spike）；
- 周期内成交量增量 vs 历史均值（volume_spike）。

表 snapshots 采用追加语义（同一 market_id 多行，按 ts 升序），而非覆盖。
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

from ..models import Market

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    market_id  TEXT NOT NULL,
    platform   TEXT NOT NULL,
    ts         INTEGER NOT NULL,
    yes_probability REAL,
    volume_total    REAL,
    volume_24h      REAL,
    liquidity       REAL,
    PRIMARY KEY (market_id, ts)
);
CREATE INDEX IF NOT EXISTS idx_snap_market ON snapshots(market_id, ts);
"""


class SnapshotStore:
    def __init__(self, db_path: str = "data/snapshots.db"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(_SCHEMA)
        # 兼容已有快照：旧行保留 NULL，不从重估后的美元金额反推合约数。
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(snapshots)")}
        for name in ("volume_contracts", "open_interest_contracts"):
            if name not in columns:
                self.conn.execute(f"ALTER TABLE snapshots ADD COLUMN {name} REAL")
        self.conn.commit()
        logger.info("快照存储就绪: %s", db_path)

    def save_snapshot(self, m: Market, ts: Optional[int] = None) -> None:
        self.save_batch([m], ts)

    def save_batch(self, markets: List[Market], ts: Optional[int] = None) -> int:
        now = ts or int(time.time())
        rows = [
            (m.market_id, m.platform, now, m.yes_probability, m.volume_total, m.volume_24h, m.liquidity,
             m.volume_contracts, m.open_interest_contracts)
            for m in markets
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO snapshots "
            "(market_id, platform, ts, yes_probability, volume_total, volume_24h, liquidity, "
            "volume_contracts, open_interest_contracts) VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def history(self, market_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        """返回某市场最近 limit 条历史快照（ts 升序）。"""
        cur = self.conn.execute(
            "SELECT ts, yes_probability, volume_total, volume_24h, liquidity, "
            "volume_contracts, open_interest_contracts FROM snapshots "
            "WHERE market_id=? ORDER BY ts DESC LIMIT ?",
            (market_id, limit),
        )
        cols = ["ts", "yes_probability", "volume_total", "volume_24h", "liquidity",
                "volume_contracts", "open_interest_contracts"]
        return [dict(zip(cols, row)) for row in reversed(cur.fetchall())]

    def previous_snapshot(self, market_id: str, before_ts: int) -> Optional[Dict[str, Any]]:
        """取 before_ts 之前最近的一条快照（用于相邻对比）。"""
        cur = self.conn.execute(
            "SELECT ts, yes_probability, volume_total, volume_24h, liquidity FROM snapshots "
            "WHERE market_id=? AND ts<? ORDER BY ts DESC LIMIT 1",
            (market_id, before_ts),
        )
        row = cur.fetchone()
        if not row:
            return None
        return dict(zip(["ts", "yes_probability", "volume_total", "volume_24h", "liquidity"], row))

    def close(self) -> None:
        self.conn.close()
