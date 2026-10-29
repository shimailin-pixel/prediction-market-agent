"""配置加载：config.yaml + .env（可选）。

- load_dotenv 存在则加载 .env，让 DASHSCOPE_API_KEY / 代理等环境变量对进程可见；
- 提供点分路径读取 cfg_get(config, "runtime.min_volume")；
- 命令行覆盖通过 apply_overrides 统一处理。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

# 项目根（src/ 的上一级）：不依赖启动时的 cwd，避免从不同目录运行时配置/数据路径漂移
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv

    _HAS_DOTENV = True
except ImportError:  # pragma: no cover
    _HAS_DOTENV = False


def resolve_config_path(path: str) -> str:
    """相对路径先按 cwd 找；不存在则回退到项目根（兼容从任意目录启动）。"""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(candidate):
        return candidate
    return path


def project_path(rel: str) -> str:
    """把配置里的相对输出/数据路径锚定到项目根，确保跨目录启动时落在同一处。"""
    return rel if os.path.isabs(rel) else os.path.join(PROJECT_ROOT, rel)


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """读取 YAML 主配置，并把 .env 中的密钥/代理注入 os.environ。"""
    if _HAS_DOTENV:
        load_dotenv(override=False)  # 不覆盖已在系统环境中的变量
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)  # 项目 .env

    path = resolve_config_path(path)
    if not os.path.exists(path):
        logger.warning("未找到 %s，使用空配置（内置默认值兜底）", path)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    logger.info("已加载配置: %s", path)
    return cfg


def cfg_get(config: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """按点分路径读取，如 cfg_get(cfg, 'runtime.min_volume', 0)。"""
    node: Any = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def get_llm_credentials(agent_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """解析 Agent 判断层所需的密钥与端点。

    优先级：显式 AGENT_* / DASHSCOPE_* / OPENAI_* 环境变量。
    返回 {api_key, base_url, model}；api_key 为空表示应跳过 LLM。
    """
    api_key = (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("AGENT_API_KEY")
        or ""
    )
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("AGENT_BASE_URL")
        or agent_cfg.get("base_url")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("AGENT_MODEL") or agent_cfg.get("model", "qwen-plus")
    return {"api_key": api_key, "base_url": base_url, "model": model}


def apply_overrides(
    config: Dict[str, Any],
    platforms: List[str] = None,
    keywords: List[str] = None,
    min_volume: float = None,
    limit: int = None,
    disable_llm: bool = False,
) -> Dict[str, Any]:
    """把命令行参数覆盖进配置（就地修改并返回）。"""
    rt = config.setdefault("runtime", {})
    if platforms:
        rt["platforms"] = platforms
    if keywords:
        rt["keywords"] = keywords
    if min_volume is not None:
        rt["min_volume"] = min_volume
    if limit is not None:
        rt["limit"] = limit
    if disable_llm:
        config.setdefault("agent", {})["enabled"] = False
    return config
