"""数据接入层：把 Polymarket / Kalshi 原始 API 归一化为统一的 Market。"""

from .polymarket import PolymarketClient
from .kalshi import KalshiClient
from .storage import SnapshotStore
from .embeddings import EmbeddingClient

__all__ = ["PolymarketClient", "KalshiClient", "SnapshotStore", "EmbeddingClient"]
