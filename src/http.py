"""极简 JSON HTTP 客户端。

设计要点（沿用 polymarket_monitor.py 的健壮性）：
- httpx 优先，缺失时自动回落标准库 urllib，避免强依赖；
- 带重试与指数退避，兼容 429 限流；
- 支持可选代理（读取 HTTP(S)_PROXY 环境变量），国内访问任一平台失败不影响主流程。
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:  # httpx 可选
    import httpx

    _HAS_HTTPX = True
except ImportError:  # pragma: no cover - 环境相关
    httpx = None  # type: ignore
    _HAS_HTTPX = False


class HttpError(Exception):
    """所有网络异常的统一封装。"""


_DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


class HttpClient:
    """带重试 / 退避 / 可选代理的 JSON GET 客户端。"""

    def __init__(
        self,
        timeout: float = 20.0,
        retries: int = 3,
        backoff: float = 1.6,
        proxy: Optional[str] = None,
    ):
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.proxy = proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

        self._session = None
        if _HAS_HTTPX:
            transport = httpx.HTTPTransport(proxy=self.proxy) if self.proxy else None
            self._session = httpx.Client(
                timeout=timeout, headers=_DEFAULT_HEADERS, transport=transport
            )

    def get_json(self, url: str, params: Optional[dict] = None) -> Any:
        """GET 并解析 JSON；重试耗尽后抛 HttpError。"""
        last_err: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                if self._session is not None:
                    resp = self._session.get(url, params=params)
                    if resp.status_code == 429:
                        time.sleep(self.backoff ** (attempt + 1))
                        continue
                    resp.raise_for_status()
                    return resp.json()
                return self._urllib_get(url, params)
            except Exception as exc:  # noqa: BLE001 - 统一转译
                last_err = exc
                if attempt < self.retries - 1:
                    time.sleep(self.backoff ** attempt)
        raise HttpError(f"{url} 请求失败: {last_err}")

    def _urllib_get(self, url: str, params: Optional[dict]) -> Any:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=_DEFAULT_HEADERS)
        if self.proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": self.proxy, "https": self.proxy})
            )
        else:
            opener = urllib.request.build_opener()
        try:
            with opener.open(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise HttpError(f"HTTP {exc.code}") from exc
        except Exception as exc:  # noqa: BLE001
            raise HttpError(str(exc)) from exc

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
