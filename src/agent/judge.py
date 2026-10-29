"""Agent 判断层：调用 LLM 对异动事件做结构化解读。

- 走 OpenAI 兼容端点（默认通义千问 DashScope），密钥从环境变量读取；
- 输入是补全后的 Anomaly（概率/变化/成交额等事实已在 prompt 里给出，模型不臆测数字）；
- 输出用 pydantic 校验（借鉴 TradingAgents 的结构化输出层）：模型只填分析字段，
  事实字段（id/平台/概率/异动类型/时间戳）由代码侧填充；
- 无 key 或调用失败时优雅降级：返回 None，主流程（数据+信号）不受影响。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict
from .validation import validate_evidence

from ..models import Anomaly
from .prompt import SYSTEM_PROMPT, JUDGE_PROMPT, build_evidence

logger = logging.getLogger(__name__)


class ImpactDirection(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    A股: str = Field(min_length=1)
    美元: str = Field(min_length=1)
    大宗商品: str = Field(min_length=1)
    黄金: str = Field(min_length=1)


class Judgment(BaseModel):
    """单个异动事件的结构化判断（事实字段代码填，分析字段模型填）。"""

    # 事实字段
    market_id: str
    platform: str
    question: str
    anomaly_type: str
    current_probability: float
    timestamp: str
    # 分析字段（来自 LLM）
    model_config = ConfigDict(str_strip_whitespace=True)
    validation_status: str = "automated_screen_only"
    event_summary: str = Field(min_length=1)
    event_category: str
    category_cn: str
    impact_on_a_share: str
    impact_direction: ImpactDirection
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    trade_bias: str
    rationale: str = Field(min_length=1)
    risk_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class EventJudge:
    def __init__(self, api_key: str, base_url: str, model: str = "qwen-plus",
                 temperature: float = 0.1, max_tokens: int = 900):
        self.validation_failures = []
        self.enabled = bool(api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        if self.enabled:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info("Agent 判断层就绪: model=%s, base_url=%s", model, base_url)
        else:
            logger.warning("未配置 LLM 密钥，Agent 判断层跳过（数据+信号不受影响）")

    def judge(self, anomaly: Anomaly) -> Optional[Judgment]:
        if self.client is None or anomaly.market is None:
            return None
        m = anomaly.market
        evidence = build_evidence(anomaly.details, anomaly.anomaly_type)
        prompt = JUDGE_PROMPT.format(
            today=datetime.now().strftime("%Y-%m-%d"),
            platform=anomaly.platform,
            question=anomaly.question,
            primary_type=m.primary_type or "未分类",
            question_shape=m.question_shape,
            anomaly_type=anomaly.anomaly_type,
            current_prob_pct=round(anomaly.current_probability * 100, 1),
            evidence=evidence,
            description=(m.description or "无")[:300],
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"} if ("gpt" in self.model or "qwen" in self.model) else None,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            judgment = Judgment(
                market_id=anomaly.market_id,
                platform=anomaly.platform,
                question=anomaly.question,
                anomaly_type=anomaly.anomaly_type,
                current_probability=anomaly.current_probability,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                event_summary=data.get("event_summary", ""),
                event_category=data.get("event_category", "Other"),
                category_cn=data.get("category_cn", ""),
                impact_on_a_share=data.get("impact_on_a_share", "中性"),
                impact_direction=ImpactDirection(**(data.get("impact_direction") or {})),
                confidence=float(data.get("confidence", 0.0)),
                trade_bias=data.get("trade_bias", "观望"),
                rationale=data.get("rationale", ""),
                risk_note=data.get("risk_note", ""),
            )
            issues = validate_evidence(data, anomaly)
            if issues:
                self.validation_failures.append({"market_id": anomaly.market_id, "question": anomaly.question, "reasons": issues})
                return None
            return judgment
        except Exception as e:  # noqa: BLE001 - 单事件失败不影响整体
            self.validation_failures.append({"market_id": anomaly.market_id, "question": anomaly.question,
                                             "reasons": ["模型调用、JSON 或必填字段校验失败"]})
            logger.error("Agent 判断失败 (%s): %s", anomaly.market_id, e)
            return None

    @staticmethod
    def select_diverse(anomalies: List[Anomaly], max_events: int,
                      min_per_platform: int = 0) -> List[Anomaly]:
        """平台多样化选取：每个平台保底 min_per_platform 个名额，剩余按全局排序补齐。

        防止 high 级全被单平台占据（如 whale 信号只有 Polymarket 有增强数据），
        导致小平台的 medium 异动永远进不了 Agent。
        """
        if min_per_platform <= 0:
            return anomalies[:max_events]
        by_plat: Dict[str, List[Anomaly]] = {}
        for a in anomalies:
            by_plat.setdefault(a.platform, []).append(a)
        selected: List[Anomaly] = []
        seen = set()
        for items in by_plat.values():
            for a in items[:min_per_platform]:
                selected.append(a)
                seen.add(id(a))
        for a in anomalies:
            if len(selected) >= max_events:
                break
            if id(a) not in seen:
                selected.append(a)
                seen.add(id(a))
        order = {id(a): i for i, a in enumerate(anomalies)}
        selected.sort(key=lambda a: order[id(a)])
        return selected[:max_events]

    def judge_batch(self, anomalies: List[Anomaly], max_events: int = 8,
                    min_per_platform: int = 0) -> List[Judgment]:
        if self.client is None:
            return []
        picked = self.select_diverse(anomalies, max_events, min_per_platform)
        judgments: List[Judgment] = []
        for a in picked:
            j = self.judge(a)
            if j:
                judgments.append(j)
        logger.info("Agent 批量判断完成: %d/%d", len(judgments), len(picked))
        return judgments
