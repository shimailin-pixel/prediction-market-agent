"""分析框架离线单元测试（不触网、不调用 LLM）。

覆盖重建后的核心逻辑：
- classify：词边界正则（消除 toward→war 子串误伤）、类型优先级、问题形态；
- signals：有价值过滤（剔除已过期/零流动性/长尾）、聪明钱资金涌入、资产建议聚合；
- cross_platform：同事件保守配对与概率背离；
- tracker：相对历史快照的概率跳变；
- util：安全类型转换与格式化。
"""

from __future__ import annotations

from src.analysis.classify import (
    annotate,
    classify_question_shape,
    type_from_keyword,
    types_from_question,
)
from src.analysis.cross_platform import match_markets
from src.analysis.elections import is_local_election, is_off_theme, is_us_state_election
from src.analysis.signals import (
    dedupe_anomalies_by_event,
    detect_smart_money,
    filter_relevant,
    filter_valuable,
    generate_asset_suggestions,
    quota_truncate,
)
from src.analysis.threshold_series import summarize as summarize_series
from src.analysis.tracker import track_changes
from src.agent.judge import EventJudge
from src.models import Anomaly, Market
from src.util import (
    days_until,
    format_volume,
    meeting_hint,
    parse_json_field,
    safe_float,
    short_question,
)

CFG = {
    "runtime": {"min_volume": 10000, "min_liquidity": 1, "suggest_top_n": 3},
    "signals": {
        "volume_multiple": 3.0,
        "volume_24h_share": 0.08,
        "prob_change_24h_pts": 5.0,
        "prob_change_7d_pts": 10.0,
        "concentration_top1_pct": 45.0,
        "prob_spike_pts": 15.0,
        "cross_platform_divergence_pts": 10.0,
        "similarity_cutoff": 0.62,
    },
}


def _mk(**kw) -> Market:
    kw.setdefault("market_id", kw.get("question", "m")[:12])
    kw.setdefault("platform", "polymarket")
    kw.setdefault("yes_probability", 0.5)
    kw.setdefault("no_probability", 1 - kw["yes_probability"])
    return Market(**kw)


# --------------------------------------------------------------------------- classify
def test_word_boundary_avoids_substring_match():
    # "toward" 含子串 "war"，词边界正则不应误判为 war 类型
    assert "war" not in types_from_question("US easing toward de-escalation")
    assert "war" in types_from_question("Will Russia start a war in 2026?")


def test_primary_type_priority():
    # ceasefire 优先级高于 iran/mideast 等地理类
    m = annotate(_mk(question="Will a ceasefire hold between Iran and Israel?"))
    assert m.primary_type == "ceasefire"
    assert "iran" in m.event_types and "mideast" in m.event_types


def test_question_shape_threshold_vs_event():
    assert classify_question_shape("Will WTI oil be above $110 on Dec 31?") == "threshold"
    assert classify_question_shape("Will Iran nuclear deal be signed in 2026?") == "event"


def test_type_from_keyword_fallback():
    assert type_from_keyword("Fed") == "interest_rate"
    assert type_from_keyword("ukraine") == "russia"
    assert type_from_keyword("nonsense") == ""


# ---------------------------------------------------------------------------- signals
def test_filter_valuable_drops_noise():
    keep = _mk(question="live market", days_to_resolution=10, liquidity=5000, volume_total=1_000_000)
    expired = _mk(question="expired", days_to_resolution=-3, liquidity=5000, volume_total=1_000_000)
    no_liq = _mk(question="dead", days_to_resolution=10, liquidity=0, volume_total=1_000_000)
    longtail = _mk(question="tiny", days_to_resolution=10, liquidity=5000, volume_total=50)
    kept = filter_valuable([keep, expired, no_liq, longtail], CFG)
    assert kept == [keep]


def test_filter_valuable_per_source_override():
    cfg_ps = {"runtime": {"min_volume": 10000, "min_liquidity": 1,
                          "per_source": {"kalshi": {"min_volume": 2000}}}}
    ks_mid = _mk(platform="kalshi", question="fed funds above 4%",
                 volume_total=5000, liquidity=500, days_to_resolution=7)
    pm_mid = _mk(platform="polymarket", question="fed rate cut",
                 volume_total=5000, liquidity=500, days_to_resolution=7)
    kept = filter_valuable([ks_mid, pm_mid], cfg_ps)
    assert ks_mid in kept      # Kalshi 用 2000 阈值，保留
    assert pm_mid not in kept  # Polymarket 仍用 10000，剔除


def test_detect_smart_money_flags_inflow():
    m = _mk(question="Iran strike risk", volume_total=1_000_000, volume_24h=200_000)
    annotate(m)
    anomalies = detect_smart_money([m], CFG)
    assert anomalies
    assert "money_inflow" in anomalies[0].details["signal_tags"]


def test_generate_asset_suggestions_orders_and_reviews_threshold():
    event_m = annotate(_mk(question="Will there be a war in region X?", yes_probability=0.85,
                           volume_total=5_000_000, volume_24h=20_000, liquidity=9999, days_to_resolution=20))
    level_m = annotate(_mk(question="Will WTI oil be above $110?", yes_probability=0.90,
                           volume_total=5_000_000, volume_24h=20_000, liquidity=9999, days_to_resolution=20))
    sug = generate_asset_suggestions([event_m, level_m], CFG)
    assert sug
    assert all(s["direction"] == "high" for s in sug if s["event_type"] in ("war", "oil"))
    level_sug = [s for s in sug if s["event_type"] == "oil"]
    assert level_sug == []  # 阈值概率不推出资产涨跌


def test_suggestions_gates_tail_lottery_and_inactive():
    cfg_g = {"runtime": {"suggest_top_n": 3, "suggest_min_volume_24h": 500,
                          "suggest_tail_prob_floor": 2.0}}
    tail = annotate(_mk(market_id="t1", question="Will the Fed cut rates by 50 bps?",
                        yes_probability=0.001, volume_total=9_000_000, volume_24h=50_000,
                        liquidity=1000, days_to_resolution=10))
    dead = annotate(_mk(market_id="d1", question="Will Kanye West perform in Russia?",
                        yes_probability=0.70, volume_total=900_000, volume_24h=0,
                        liquidity=1000, days_to_resolution=10))
    live = annotate(_mk(market_id="l1", question="Will there be a war next year?",
                        yes_probability=0.70, volume_total=900_000, volume_24h=5000,
                        liquidity=1000, days_to_resolution=10))
    sug = generate_asset_suggestions([tail, dead, live], cfg_g)
    triggers = {s["trigger"] for s in sug}
    assert not any("50 bps" in t for t in triggers), "0.1% 尾概率彩票不应出建议"
    assert not any("Kanye" in t for t in triggers), "24h 零成交的噱头盘不应出建议"
    assert any("war" in t.lower() for t in triggers), "活跃高信念盘应正常出建议"


def test_suggestions_negation_rate_market_flips_bucket():
    # 「no Fed rate cuts 96%」= 市场认为不降息（偏鹰），不能归入 low=降息预期桶
    cfg_g = {"runtime": {"suggest_top_n": 3, "suggest_min_volume_24h": 0}}
    no_cuts = annotate(_mk(market_id="nc", question="Will no Fed rate cuts happen in 2026?",
                           yes_probability=0.96, volume_total=9_000_000, volume_24h=20_000,
                           liquidity=1000, days_to_resolution=30))
    sug = generate_asset_suggestions([no_cuts], cfg_g)
    assert sug == []  # 不降息不等于加息


def test_suggestions_geopolitical_low_prob_needs_review():
    # 入侵盘低概率 = 现状推论，不是缓和事件，建议必须标需复核
    cfg_g = {"runtime": {"suggest_top_n": 3, "suggest_min_volume_24h": 0,
                          "suggest_tail_prob_floor": 1.0}}
    war_m = annotate(_mk(market_id="w1", question="Will China invade Taiwan by end of 2027?",
                         yes_probability=0.03, volume_total=45_000_000, volume_24h=200_000,
                         liquidity=1000, days_to_resolution=60))
    sug = generate_asset_suggestions([war_m], cfg_g)
    war_sug = [s for s in sug if s["event_type"] == "war"]
    assert war_sug == []  # 风险未发生不等于风险资产上涨


def test_suggestions_aggregate_same_event():
    # 同一事件（event_key 相同）的 3 个互斥合约只出 1 条建议
    cfg_g = {"runtime": {"suggest_top_n": 3, "suggest_min_volume_24h": 0,
                          "suggest_tail_prob_floor": 0.0}}
    ms = [annotate(_mk(market_id=f"iran{i}", event_key="iran-head-2026",
                       question=f"Will there be a war in region {i}?",
                       yes_probability=0.80, volume_total=(3 - i) * 1_000_000,
                       liquidity=1000, days_to_resolution=30)) for i in range(3)]
    sug = generate_asset_suggestions(ms, cfg_g)
    iran_sug = [s for s in sug if s["event_type"] == "war"]
    assert len(iran_sug) == 1, f"同事件应聚合为 1 条，实际 {len(iran_sug)}"
    assert "region 0" in iran_sug[0]["trigger"]  # 代表 = volume 最大者


# ------------------------------------------------------------------------ cross_platform
def test_cross_platform_pair_and_divergence():
    pm = annotate(_mk(platform="polymarket", question="Will the Fed cut interest rates in September?",
                      yes_probability=0.70))
    ks = annotate(_mk(platform="kalshi", question="Will the Fed cut interest rates in September?",
                      yes_probability=0.50, ticker="KXFED"))
    anomalies, pairs = match_markets([pm, ks], similarity_cutoff=0.62, divergence_pts=10.0)
    assert pairs, "同措辞市场应被配对"
    assert anomalies and anomalies[0].anomaly_type == "cross_platform_divergence"
    assert anomalies[0].details["divergence_pts"] == 20.0


def test_cross_platform_respects_type_gate():
    pm = annotate(_mk(platform="polymarket", question="Will the Fed cut rates in September?", yes_probability=0.7))
    ks = annotate(_mk(platform="kalshi", question="Will WTI oil be above 100 in September?",
                      yes_probability=0.5, ticker="KOIL"))
    _, pairs = match_markets([pm, ks], similarity_cutoff=0.5, divergence_pts=1.0)
    assert pairs == []  # 不同 primary_type 不配对


class _FakeEmbedder:
    """离线 mock 向量后端：不触网验证 embedding 匹配路径与降级。"""

    enabled = True

    def __init__(self, vectors, fail=False):
        self.vectors = vectors
        self.fail = fail

    def embed(self, texts):
        if self.fail:
            return None
        return {t: self.vectors[t] for t in texts if t in self.vectors}


def test_cross_platform_embedding_rejects_non_equivalent_propositions():
    # 同主题但非等价的命题，即使 embedding 分数很高也必须拒配
    pm = annotate(_mk(platform="polymarket",
                      question="Will there be no change in Fed interest rates after the October meeting?",
                      yes_probability=0.9))
    ks = annotate(_mk(platform="kalshi",
                      question="Will the upper bound of the federal funds rate be above 3.75% following the Fed's October meeting?",
                      yes_probability=0.6, ticker="KXFED-U375"))
    vectors = {pm.question: [1.0, 0.0, 0.0], ks.question: [0.99, 0.14, 0.0]}
    anomalies, pairs = match_markets([pm, ks], embedder=_FakeEmbedder(vectors),
                                     embedding_cutoff=0.80, divergence_pts=10.0)
    assert pairs == [] and anomalies == []
    # 回归对照：同样两个市场，纯文本模式配不上
    _, plain_pairs = match_markets([pm, ks], similarity_cutoff=0.62)
    assert plain_pairs == []


def test_cross_platform_embedding_failure_falls_back_to_text():
    pm = annotate(_mk(platform="polymarket", question="Will the Fed cut interest rates in September?",
                      yes_probability=0.7))
    ks = annotate(_mk(platform="kalshi", question="Will the Fed cut interest rates in September?",
                      yes_probability=0.5, ticker="KXFED2"))
    anomalies, pairs = match_markets([pm, ks], embedder=_FakeEmbedder({}, fail=True),
                                     similarity_cutoff=0.62, divergence_pts=10.0)
    # embed 失败返回 None → 降级文本，同措辞仍应配对且 match_method=text
    assert pairs
    assert anomalies[0].details["match_method"] == "text"


def test_cross_platform_central_bank_subject_conflict():
    # 报告实跑暴露的误配：Fed 加息 vs 日本央行加息，同 type 且语义向量高分，必须拒配
    pm = annotate(_mk(platform="polymarket",
                      question="Will the Fed increase interest rates by 25 bps after the October meeting?",
                      yes_probability=0.765))
    ks = annotate(_mk(platform="kalshi",
                      question="Will the Bank of Japan Hike 25bps at the October meeting?",
                      yes_probability=0.715, ticker="KXBOJ"))
    vectors = {pm.question: [1.0, 0.0], ks.question: [1.0, 0.01]}  # 强行给高分绕过 cutoff
    anomalies, pairs = match_markets([pm, ks], embedder=_FakeEmbedder(vectors),
                                     embedding_cutoff=0.80, divergence_pts=10.0)
    assert pairs == [] and anomalies == [], "不同央行主体宁漏不假配"


def test_cross_platform_low_divergence_pair_dropped():
    # 背离不足门禁阈值的配对不进报告（三连门禁第三道）
    pm = annotate(_mk(platform="polymarket", question="Will the Fed cut interest rates in September?",
                      yes_probability=0.60))
    ks = annotate(_mk(platform="kalshi", question="Will the Fed cut interest rates in September?",
                      yes_probability=0.55, ticker="KXFED3"))
    anomalies, pairs = match_markets([pm, ks], similarity_cutoff=0.62, divergence_pts=10.0)
    assert pairs == [], "背离 5 点 < 门禁 10 点，不应展示"
    assert anomalies == []


# ------------------------------------------------------------------------------- tracker
class _FakeStore:
    def __init__(self, rows):
        self._rows = rows

    def history(self, market_id):
        return self._rows


def test_track_changes_prob_spike():
    m = _mk(question="live market", yes_probability=0.80)
    prior_ts = 1_000_000
    store = _FakeStore([{"ts": prior_ts, "yes_probability": 0.55, "volume_total": 1000}])
    anomalies = track_changes([m], store, CFG, now_ts=prior_ts + 3600)
    assert anomalies and anomalies[0].anomaly_type == "prob_spike"
    assert anomalies[0].details["change_pts"] == 25.0


def test_track_changes_first_run_no_history():
    m = _mk(question="brand new", yes_probability=0.6)
    anomalies = track_changes([m], _FakeStore([]), CFG, now_ts=2_000_000)
    assert anomalies == []


# ------------------------------------------------------------------------------- util
def test_safe_float_tolerant():
    assert safe_float("$1,234.5") == 1234.5
    assert safe_float(None) == 0.0
    assert safe_float("n/a", default=-1) == -1
    assert safe_float(3) == 3.0


def test_short_question_keeps_tail_for_ladder_titles():
    q = ("Will the upper bound of the federal funds rate be above 4.25% following "
         "the Federal Reserve's March 2027 meeting?")
    s = short_question(q, 72)
    assert len(s) <= 72 and s != q[:72]
    assert "4.25%" in s.split("…")[0] or "2027" in s.split("…")[-1], "区分信息（阈值/会议）必须保留"
    # 短标题不受影响
    assert short_question("Will Iran change leadership?", 72) == "Will Iran change leadership?"


def test_meeting_hint_extracts_case_tail():
    q = "Will the upper bound of the federal funds rate be above 3.75% following the Fed's December meeting?"
    assert meeting_hint(q) == "the Fed's December meeting" or "December meeting" in meeting_hint(q)
    assert meeting_hint("Will Iran have a new leader?") == ""


def test_parse_json_field():
    assert parse_json_field('["a", "b"]') == ["a", "b"]
    assert parse_json_field("not json", default=[]) == []
    assert parse_json_field({"k": 1}) == {"k": 1}


def test_format_volume_and_days_until():
    assert format_volume(1_200_000_000) == "$1.2B"
    assert format_volume(3_400_000) == "$3.4M"
    assert format_volume(5_600) == "$5.6K"
    assert format_volume(7) == "$7"
    assert isinstance(days_until("2999-01-01T00:00:00Z"), int)
    assert days_until("") is None


# ------------------------------------------------------------------ elections (州级选举剔除)
def test_elections_state_level_detected():
    for q in [
        "Will Republicans win the Senate race in Texas?",
        "Will Democratics win the Senate race in Michigan?",
        "Will Todd Achilless win the Senate race in Idaho?",
        "Who will win the Texas House of Representatives?",
        "Will Democrats win the governor's race in California?",
    ]:
        assert is_us_state_election(q), q


def test_elections_national_preserved():
    for q in [
        "Will Trump win the 2028 presidential election?",
        "Who will control the US Senate after the midterms?",
        "Will the Republicans win the House of Representatives nationally?",
    ]:
        assert not is_us_state_election(q), q


def test_elections_no_geo_false_positive():
    # 州名作为地理/地缘实体出现，但无选举语境 → 不误伤
    for q in [
        "Will Russia invade Georgia?",
        "Will the US strike Iran from Washington bases?",
        "Will oil spill near California coast?",
    ]:
        assert not is_us_state_election(q), q


# ------------------------------------------------------- 相关性门禁（关键词 + 体育娱乐剔除）
def test_off_theme_sports_entertainment():
    for q in [
        "Will more than 600 games be played in the 2027 Pro Baseball regular season?",
        "Will LeBron James announce his retirement before the 2027-28 NBA season?",
        "Will Ella Langley win Entertainer of the Year at the CMA Awards?",
        "Will LUX win Album of the Year at the 2026 Latin Grammy Awards?",
    ]:
        assert is_off_theme(q), q
    for q in [
        "Will the Fed cut interest rates in December?",
        "Will there be a ceasefire in Gaza by 2026?",
    ]:
        assert not is_off_theme(q), q


def test_filter_relevant_keyword_gate():
    cfg_rel = {"runtime": {"keywords": ["Fed", "Iran"], "require_keyword_match": True}}
    fed = annotate(_mk(question="Will the Fed cut rates in December?"))
    iran = annotate(_mk(question="Will the US strike Iran before 2027?"))
    baseball = annotate(_mk(question="Will more than 1,930 games be played in the 2027 season?"))
    bulgaria = annotate(_mk(question="Will Iliana Yotova win the 2026 Bulgarian presidential election?"))
    kept = filter_relevant([fed, iran, baseball, bulgaria], cfg_rel)
    ids = {k.market_id for k in kept}
    assert fed.market_id in ids and iran.market_id in ids
    assert baseball.market_id not in ids and bulgaria.market_id not in ids


def test_filter_relevant_empty_keywords_keeps_taxonomy_only():
    cfg_rel = {"runtime": {"keywords": []}}
    macro = annotate(_mk(question="Will the US enter a recession in 2026?"))
    sports = annotate(_mk(question="Who will win the 2027 Super Bowl?"))
    random = annotate(_mk(question="Will it rain in Paris on Nov 1?"))
    kept = filter_relevant([macro, sports, random], cfg_rel)
    assert kept == [macro]


# --------------------------------------------------------------------------- 平台配额


def test_quota_truncate_guarantees_small_platform():
    pool = [
        _mk(market_id=f"p{i}", platform="polymarket", question=f"pm{i}",
            volume_total=1_000_000 - i, liquidity=100, days_to_resolution=5)
        for i in range(20)
    ] + [
        _mk(market_id=f"k{i}", platform="kalshi", question=f"ks{i}",
            volume_total=50_000 - i, liquidity=100, days_to_resolution=5)
        for i in range(5)
    ]
    out = quota_truncate(pool, limit=10, min_per_platform=4)
    assert len(out) <= 10
    assert sum(1 for m in out if m.platform == "kalshi") >= 4
    # 无配额时 Kalshi 被全盘吞掉（回归原来的量级淹没问题）
    plain = quota_truncate(pool, limit=10, min_per_platform=0)
    assert all(m.platform == "polymarket" for m in plain)


def test_smart_money_volume_spike_uses_per_platform_baseline():
    pm = [
        _mk(market_id=f"pm{i}", platform="polymarket", question=f"x{i}",
            volume_total=2_000_000, volume_24h=1000, liquidity=100, days_to_resolution=10)
        for i in range(3)
    ]
    ks_small = [
        _mk(market_id=f"ks{i}", platform="kalshi", question=f"y{i}",
            volume_total=500, volume_24h=10, liquidity=100, days_to_resolution=10)
        for i in range(3)
    ]
    ks_star = _mk(market_id="ks_star", platform="kalshi", question="big spike",
                  volume_total=30_000, volume_24h=100, liquidity=100, days_to_resolution=10)
    anoms = detect_smart_money(pm + ks_small + [ks_star], CFG)
    hit = [a for a in anoms
           if a.market_id == "ks_star" and "volume" in a.details["signal_tags"]]
    # 旧逻辑用全体均值（~86万），3× 均值 ≈ 259万，3万的 Kalshi 盘永远不会报；
    # 新逻辑用本平台均值（~5583），3× ≈ 1.67万 → 命中
    assert hit, "Kalshi 放量盘应基于本平台均值触发 volume 信号"
    assert all(a.platform != "kalshi" or "money_inflow" in a.details["signal_tags"]
               or "volume" in a.details["signal_tags"] for a in anoms)


def test_select_diverse_reserves_platform_slots():
    def _a(i, plat):
        return Anomaly(market_id=f"{plat}{i}", platform=plat, question=f"q{i}",
                       anomaly_type="volume",
                       severity="high" if plat == "polymarket" else "medium",
                       current_probability=0.5)
    anoms = [_a(i, "polymarket") for i in range(8)] + [_a(i, "kalshi") for i in range(3)]
    picked = EventJudge.select_diverse(anoms, max_events=8, min_per_platform=2)
    assert len(picked) == 8
    assert sum(1 for a in picked if a.platform == "kalshi") >= 2
    # 不设配额时回归原始行为：high 全被 PM 占据，Kalshi 一条进不了
    plain = EventJudge.select_diverse(anoms, 8, 0)
    assert all(a.platform == "polymarket" for a in plain)


# ------------------------------------------------------------------------ 事件级去重 / 阈值系列
def _fed_ladder():
    """同一 Kalshi 事件（event_ticker 相同）的 4 个互斥阈值合约。"""
    rows = [("above 3.25%", 0.985, 30_000), ("above 3.75%", 0.76, 40_000),
            ("above 4.25%", 0.205, 20_000), ("above 4.75%", 0.02, 10_000)]
    return [
        _mk(market_id=f"KXFED{i}", platform="kalshi", event_key="KXFEDDECISION-26DEC",
            question=f"Will the upper bound of the federal funds rate be above {q.split('above ')[1]} in December?",
            yes_probability=p, volume_total=v, liquidity=5000, days_to_resolution=60)
        for i, (q, p, v) in enumerate(rows)
    ]


def test_dedupe_anomalies_by_event_keeps_one_rep_per_event():
    ladder = _fed_ladder()
    anoms = [
        Anomaly(market_id=m.market_id, platform=m.platform, question=m.question,
                anomaly_type="volume", severity="high", current_probability=m.yes_probability,
                details={"volume_total": m.volume_total}, market=m)
        for m in ladder
    ]
    # 外加一个无 event_key 的异动（不参与聚合）
    stray = Anomaly(market_id="s1", platform="polymarket", question="other",
                    anomaly_type="whale", severity="medium", current_probability=0.5, details={})
    kept = dedupe_anomalies_by_event(anoms + [stray], per_event_max=1)
    ks_kept = [a for a in kept if a.platform == "kalshi"]
    assert len(ks_kept) == 1, "同事件 4 个阶梯合约只应留 1 个代表"
    assert ks_kept[0].market_id == "KXFED1", "代表应为同级 severity 中累计成交额最大者"
    assert stray in kept and len(kept) == 2
    # per_event_max=0 回归不去重
    assert len(dedupe_anomalies_by_event(anoms, 0)) == 4


def test_is_local_election():
    assert is_local_election("Will Puma Shen win the next Taipei mayoral election?")
    assert is_local_election("Will the Republicans win back the state assembly?")
    # 全国性/央行人事不应误伤
    assert not is_local_election("Will Trump win the 2028 presidential election?")
    assert not is_local_election("Will Jerome Powell be removed as Fed chair by 2026?")
    assert not is_local_election("Will a Fed governor be replaced before 2027?")


def test_threshold_series_median_interpolation():
    ladder = _fed_ladder()
    summaries = summarize_series(ladder)
    assert len(summaries) == 1
    s = summaries[0]
    assert s["platform"] == "kalshi" and s["n"] == 4 and s["event_key"] == "KXFEDDECISION-26DEC"
    # above 型：0.76@3.75 → 0.205@4.25 之间穿越 0.5，线性插值 ≈ 4.027
    assert 3.75 < s["median"] < 4.25
    assert "隐含分布" in s["line"] and "个阈值合约）" in s["line"]
    # 不足 min_members 不成组
    assert summarize_series(ladder[:2]) == []
