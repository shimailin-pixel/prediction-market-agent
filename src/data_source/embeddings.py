"""批量文本向量化客户端（DashScope OpenAI 兼容 /embeddings 端点）。

- 与 Agent 判断层同模式：有密钥且 openai SDK 可用才启用，否则 enabled=False；
- 调用方（cross_platform 匹配）在 embed 返回 None 时降级回文本相似度，
  网络失败/限额/无 key 都不会打断主流程；
- DashScope text-embedding-v3 单批上限 10 条，分批请求并做简单限速。
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_BATCH_SIZE = 10  # DashScope text-embedding-v3 单次请求条数上限


class EmbeddingClient:
    def __init__(self, api_key: str, base_url: str,
                 model: str = "text-embedding-v3"):
        self.enabled = bool(api_key)
        self.model = model
        self.client = None
        if self.enabled:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=api_key, base_url=base_url)
            except ImportError:  # pragma: no cover - 环境相关
                self.enabled = False
                logger.warning("openai SDK 不可用，跨平台匹配降级为文本相似度")

    def embed(self, texts: List[str]) -> Optional[Dict[str, List[float]]]:
        """texts → {原文: 向量}；任何失败返回 None（调用方负责降级）。"""
        if self.client is None or not texts:
            return None
        out: Dict[str, List[float]] = {}
        try:
            for i in range(0, len(texts), _BATCH_SIZE):
                batch = texts[i:i + _BATCH_SIZE]
                resp = self.client.embeddings.create(model=self.model, input=batch)
                for t, item in zip(batch, resp.data):
                    out[t] = item.embedding
        except Exception as exc:  # noqa: BLE001 - 降级而非中断
            logger.warning("Embedding 调用失败（%s），降级为文本相似度", exc)
            return None
        logger.info("Embedding 就绪: %d/%d 个标题, model=%s", len(out), len(texts), self.model)
        return out


def cosine(a: List[float], b: List[float]) -> float:
    """余弦相似度；零向量返回 0。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)
