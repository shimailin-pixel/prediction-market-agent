"""通用工具：安全类型转换、JSON 解包、日期与格式化。

被 data_source（解析原始字段）与 analysis（计算指标）共用，避免重复。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Optional


def safe_float(value: Any, default: float = 0.0) -> float:
    """宽容浮点转换：处理 None、空串、带逗号/美元符号的数字串、字符串定点。"""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("$", "")
    if not text or text.lower() in {"none", "null", "n/a"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_json_field(value: Any, default: Any = None) -> Any:
    """解包「JSON 字符串套 JSON」字段（Gamma 常见）。"""
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return default
    return default


def _to_datetime(raw: Any) -> Optional[datetime]:
    """把多种时间格式解析为 tz-aware UTC datetime；失败返回 None。"""
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # Unix 秒（Kalshi 某些字段 / 纯数字串）
    if text.isdigit():
        return datetime.fromtimestamp(int(text), tz=timezone.utc)
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(str(raw).replace("Z", ""), fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def days_until(raw: Any) -> Optional[int]:
    """距离结算多少天（可为负=已过期）；解析失败返回 None。"""
    dt = _to_datetime(raw)
    if dt is None:
        return None
    return (dt - datetime.now(timezone.utc)).days


def format_volume(volume: float) -> str:
    """把金额格式化为 $1.2B / $3.4M / $5.6K / $7。"""
    if volume >= 1_000_000_000:
        return f"${volume / 1_000_000_000:.1f}B"
    if volume >= 1_000_000:
        return f"${volume / 1_000_000:.1f}M"
    if volume >= 1_000:
        return f"${volume / 1_000:.1f}K"
    return f"${volume:.0f}"


_MEETING_HINT = re.compile(
    r"\bfollowing\s+(?:the\s+)?[^,.?]*?\b(?:meeting|decision)\b",
    re.IGNORECASE,
)


def meeting_hint(question: str) -> str:
    """从标题抽会议/决议案尾巴（如 'following the Fed's December meeting'），找不到返回空串。

    Kalshi 阶梯阈值盘同名部分极长，区分信息（阈值 + 会议时间）全在尾部，
    头部定长截断会让同系列不同合约显示成一模一样的标题。
    """
    mt = _MEETING_HINT.search(question or "")
    return mt.group(0) if mt else ""


def short_question(question: str, limit: int = 72) -> str:
    """定长截断，超长时保留尾部：头…尾拼接，避免关键区分信息被截掉。"""
    q = (question or "").strip()
    if len(q) <= limit:
        return q
    tail = q[-24:].lstrip()
    head = q[: max(8, limit - 1 - len(tail))].rstrip()
    return f"{head}…{tail}"
