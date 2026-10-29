"""跨平台背离：先做保守命题门禁，再使用文本/embedding 相似度排名。

只接受经过有限同义词归一化后相同的标题及兼容的到期日期。
未知改写宁可漏配；相似度不能证明不同阈值、时间、方向的命题等价。
标题门禁不能替代人工核对平台完整结算规则，输出仍需复核。
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from ..data_source.embeddings import cosine
from ..models import Anomaly, Market
from ..util import _to_datetime

logger = logging.getLogger(__name__)

_STOP = {
    "will", "the", "a", "an", "of", "to", "in", "on", "by", "is", "be",
    "and", "or", "for", "at", "as", "with", "after", "before", "until", "through",
}

# 主体冲突组：embedding/文本相似会把「加息/利率」类不同央行的盘误配
#（如 Fed 加息 vs 日本央行加息，字面相似但完全两回事），命中两组及以上直接拒配。
_SUBJECT_CONFLICTS = (
    ("fed", re.compile(r"\b(fed|fomc|federal reserve|federal funds)\b")),
    ("boj", re.compile(r"\b(bank of japan|boj)\b")),
    ("ecb", re.compile(r"\becb\b")),
    ("boe", re.compile(r"\bbank of england\b")),
    ("snb", re.compile(r"\bswiss national bank\b")),
    ("pbo", re.compile(r"\b(people's bank of china|people's bank|pboc)\b")),
    ("rbi", re.compile(r"\b(reserve bank of india|rbi)\b")),
    ("boc", re.compile(r"\bbank of canada\b")),
    ("anz", re.compile(r"\b(rba|reserve bank of australia|australia)\b")),
)


def _subject_conflict(a: str, b: str) -> bool:
    """两侧合并命中的主体组 >1 即冲突（如一侧 Fed 另一侧 BoJ）。"""
    text = f"{(a or '')} {(b or '')}".lower()
    hit = sum(1 for _, pat in _SUBJECT_CONFLICTS if pat.search(text))
    return hit > 1


def _normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    tokens = [t for t in text.split() if t not in _STOP and len(t) > 1]
    return " ".join(sorted(tokens))


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _proposition_title(text: str) -> tuple:
    """保留词序、数字、单位、否定词及时间关系；仅折叠有限同义写法。"""
    text = text.lower().replace("’", "'")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    for pattern, replacement in (
        (r"\bfederal reserve(?:'s)?\b", "fed"),
        (r"\bfed's\b", "fed"),
        (r"\b(?:decrease|decreases|lower|lowers|cuts)\b", "cut"),
        (r"\b(?:increase|increases|raise|raises|hikes)\b", "hike"),
        (r"\binterest rates\b", "rates"),
        (r"\b(?:basis points?|bps)\b", "bp"),
        (r"\bgreater than\b|\bhigher than\b|\bover\b", "above"),
        (r"\bless than\b|\blower than\b|\bunder\b", "below"),
    ):
        text = re.sub(pattern, replacement, text)
    # 不丢弃 before/after/by/on/no，也不删除单个数字和货币/百分号。
    return tuple(t for t in re.findall(r"\d+(?:\.\d+)?|[^\W\d_]+|[$%<>=+\-≥≤]", text)
                 if t not in {"will", "the", "?"})


def _comparable(a: Market, b: Market) -> bool:
    if a.question_shape != b.question_shape or a.question_shape == "unknown":
        return False
    if _subject_conflict(a.question, b.question):
        return False
    title = _proposition_title(a.question)
    if not title or title != _proposition_title(b.question):
        return False
    if bool(a.end_date) != bool(b.end_date):
        return False
    if a.end_date:
        try:
            da, db = _to_datetime(a.end_date), _to_datetime(b.end_date)
        except (ValueError, OverflowError, OSError):
            return False
        if da is None or db is None or da.date() != db.date():
            return False
    return True


def match_markets(
    markets: List[Market],
    similarity_cutoff: float = 0.62,
    divergence_pts: float = 10.0,
    embedder=None,
    embedding_cutoff: float = 0.80,
    shape_penalty: float = 0.05,
) -> Tuple[List[Anomaly], List[Tuple[Market, Market, float]]]:
    """返回 (背离异动, 匹配对明细)。

    匹配对仅包含过命题门禁且超过背离阈值的候选，结算规则仍需人工复核。
    shape_penalty 仅为兼容旧调用保留；不同形态现在直接拒配。
    embedder 为 None / 未启用 / 调用失败时，降级为归一化文本相似度匹配。
    """
    pm = [m for m in markets if m.platform == "polymarket" and m.primary_type]
    ks = [m for m in markets if m.platform == "kalshi" and m.primary_type]
    if not pm or not ks:
        return [], []

    # 语义向量：一次性批量嵌入候选池去重后的标题（候选池已被 runtime.limit 控住规模）
    vectors: Optional[dict] = None
    if embedder is not None and getattr(embedder, "enabled", False):
        texts = sorted({m.question for m in pm} | {m.question for m in ks})
        vectors = embedder.embed(texts)
        if vectors is None:
            logger.info("Embedding 不可用，跨平台匹配降级为文本相似度")
    _norm_cache = {b.question: " ".join(_proposition_title(b.question)) for b in ks}

    anomalies: List[Anomaly] = []
    pairs: List[Tuple[Market, Market, float]] = []
    used_ks: set = set()
    near_miss: List[float] = []  # 差点对上线的分数，供实跑校准 cutoff

    for a in pm:
        na = " ".join(_proposition_title(a.question))
        va = vectors.get(a.question) if vectors is not None else None
        best, best_score = None, 0.0
        for b in ks:
            if b.ticker in used_ks or b.primary_type != a.primary_type:
                continue
            if not _comparable(a, b):
                continue
            if vectors is not None:
                vb = vectors.get(b.question)
                if not vb or not va:
                    continue
                score, cutoff = cosine(va, vb), embedding_cutoff
                method = "embedding"
            else:
                score, cutoff = _similarity(na, _norm_cache[b.question]), similarity_cutoff
                method = "text"
            if score >= cutoff and score > best_score:
                best, best_score = b, score
            elif 0.0 < cutoff - score <= 0.10:
                near_miss.append(round(score, 3))
        if best is None:
            continue

        gap_pts = abs(a.yes_probability - best.yes_probability) * 100
        # 三连门禁第三道：背离不足阈值的配对不进报告（展示层不再放宽），
        # 避免弱背离噪音淹没真正可交易的跨平台信号
        if gap_pts < divergence_pts:
            continue

        used_ks.add(best.ticker)
        pairs.append((a, best, best_score))
        higher = "Polymarket" if a.yes_probability > best.yes_probability else "Kalshi"
        anomalies.append(Anomaly(
            market_id=f"{a.market_id}|{best.ticker}",
            platform="cross-platform",
            question=a.question,
            anomaly_type="cross_platform_divergence",
            severity="high",
            current_probability=a.yes_probability,
            details={
                "polymarket_id": a.market_id,
                "polymarket_prob": round(a.yes_probability, 4),
                "kalshi_ticker": best.ticker,
                "kalshi_question": best.question,
                "kalshi_prob": round(best.yes_probability, 4),
                "divergence_pts": round(gap_pts, 1),
                "higher_platform": higher,
                "similarity": round(best_score, 3),
                "match_method": method,
                "needs_review": True,
                "comparison_basis": "conservative_title_and_date",
            },
            market=a,
        ))

    mode = "embedding" if vectors is not None else "text"
    logger.info("跨平台匹配(%s): %d 对同事件, %d 对背离超阈值", mode, len(pairs), len(anomalies))
    if near_miss:
        logger.info("配对临界分数（未过线 Top5）: %s —— 可用于校准 cutoff",
                    sorted(near_miss, reverse=True)[:5])
    return anomalies, pairs
