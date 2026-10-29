#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prediction Market Agent — 主入口。

流水线（数据源 + 分析框架 + Agent 判断层）：
  采集(Polymarket/Kalshi) → 分类 → 有价值过滤 → 增强(概率历史/持仓) →
  聪明钱信号 + 动态异动追踪 + 跨平台背离 → Agent 判断 → 资产建议 → 落快照 → 输出。

用法：
  python main.py                                  # 默认双平台、内置关键词
  python main.py --keywords Iran,Fed,tariff        # 自定义关键词
  python main.py --platforms polymarket            # 只跑一个平台
  python main.py --no-llm                          # 跳过 Agent 判断（仅数据+信号）
  python main.py --no-enrich                       # 跳过 CLOB/Data 增强（更快）
  python main.py --min-volume 50000 --limit 200
  python main.py --selftest                         # 离线自检
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import List

from src.config import load_config, cfg_get, get_llm_credentials, apply_overrides, project_path
from src.models import Market
from src.data_source import PolymarketClient, KalshiClient, SnapshotStore, EmbeddingClient
from src.analysis import classify, signals, tracker, cross_platform, elections, threshold_series
from src.agent.judge import EventJudge
from src import report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def fetch_all(cfg):
    """从启用的平台采集市场。返回 (markets, platforms_used)。"""
    platforms = cfg_get(cfg, "runtime.platforms", ["polymarket", "kalshi"])
    keywords = cfg_get(cfg, "runtime.keywords", [])
    markets = []

    if "polymarket" in platforms:
        client = PolymarketClient(cfg)
        try:
            markets += client.fetch_markets(keywords, cfg)
        finally:
            client.close()
    if "kalshi" in platforms:
        client = KalshiClient(cfg)
        try:
            markets += client.fetch_markets(keywords, cfg)
        finally:
            client.close()

    return markets, platforms


def enrich_polymarket(markets, cfg, client):
    """对 Polymarket 成交额前 N 的市场补概率历史 + 持仓集中度。"""
    n = int(cfg_get(cfg, "runtime.enrich_top_n", 12))
    pm = [m for m in markets if m.platform == "polymarket" and m.token_id]
    pm.sort(key=lambda m: m.volume_total, reverse=True)
    done = 0
    for m in pm[:n]:
        hist = client.fetch_prob_history(m.token_id)
        if hist:
            m.prob_change_24h_pts = hist["change_24h_pts"]
            m.prob_change_7d_pts = hist["change_7d_pts"]
        conc = client.fetch_holder_concentration(m.condition_id)
        if conc:
            m.concentration = conc
        done += 1
    logger.info("增强完成: %d 个 Polymarket 市场补概率历史/持仓", done)


def run_pipeline(cfg, do_enrich=True, do_llm=True):
    now_ts = int(datetime.now().timestamp())

    # 1. 采集
    markets, platforms = fetch_all(cfg)
    logger.info("采集 %d 个市场（平台: %s）", len(markets), ",".join(platforms))

    # 2. 分类
    for m in markets:
        classify.annotate(m)

    # 2.5 剔除美国国内州级选举盘 + 各国次国家级选举盘（对 A 股无传导路径）
    if cfg_get(cfg, "runtime.exclude_us_state_elections", True):
        before = len(markets)
        markets = [m for m in markets if not elections.is_us_state_election(m.question)]
        dropped = before - len(markets)
        if dropped:
            logger.info("剔除美国州级选举市场: %d 个（剩余 %d）", dropped, len(markets))
    if cfg_get(cfg, "runtime.exclude_local_elections", True):
        before = len(markets)
        markets = [m for m in markets if not elections.is_local_election(m.question)]
        dropped = before - len(markets)
        if dropped:
            logger.info("剔除次国家级选举市场: %d 个（剩余 %d）", dropped, len(markets))

    # 2.6 关键词相关性门禁：体育/娱乐/奖项无条件剔除，其余必须命中关键词/政经分类体系
    markets = signals.filter_relevant(markets, cfg)

    # 3. 有价值过滤 + 截断（每平台保底配额，防止 Kalshi 被 Polymarket 量级淹没）
    markets = signals.filter_valuable(markets, cfg)
    markets = signals.quota_truncate(
        markets,
        int(cfg_get(cfg, "runtime.limit", 300)),
        int(cfg_get(cfg, "runtime.min_per_platform", 20)),
    )

    # 4. 增强（Polymarket 概率历史 / 持仓）
    if do_enrich and "polymarket" in platforms:
        pc = PolymarketClient(cfg)
        try:
            enrich_polymarket(markets, cfg, pc)
        finally:
            pc.close()

    # 5. 信号与异动
    anomalies = []
    anomalies += signals.detect_smart_money(markets, cfg)
    store = SnapshotStore(project_path(cfg_get(cfg, "storage.sqlite_path", "data/snapshots.db")))
    try:
        anomalies += tracker.track_changes(markets, store, cfg, now_ts)
    finally:
        store.close()

    sc = cfg.get("signals", {})
    # 跨平台配对：有密钥时优先 embedding 语义匹配，失败/无密钥降级文本相似度
    embedder = None
    if bool(sc.get("use_embedding", True)):
        ec = get_llm_credentials(cfg.get("agent", {}))
        if ec["api_key"]:
            embedder = EmbeddingClient(ec["api_key"], ec["base_url"],
                                       cfg_get(cfg, "agent.embed_model", "text-embedding-v3"))
    cp_anomalies, cp_pairs = cross_platform.match_markets(
        markets,
        similarity_cutoff=float(sc.get("similarity_cutoff", 0.62)),
        divergence_pts=float(sc.get("cross_platform_divergence_pts", 10.0)),
        embedder=embedder,
        embedding_cutoff=float(sc.get("embedding_cutoff", 0.80)),
        shape_penalty=float(sc.get("shape_mismatch_penalty", 0.05)),
    )
    anomalies += cp_anomalies

    # 5.5 阈值系列聚合（Kalshi Fed 利率阶梯等）：折成隐含分布摘要，
    # 并作为结构化证据附到同系列市场的异动 details（Agent 拿到完整分布而非单点）
    ts_summaries = threshold_series.summarize(markets)
    if ts_summaries:
        by_key = {}
        for s in ts_summaries:
            key = (s["platform"], s["event_key"])
            by_key[key] = by_key.get(key, "") + s["line"] + "\n"
        for a in anomalies:
            if a.market:
                line = by_key.get((a.market.platform, a.market.event_key or a.market.market_id))
                if line:
                    a.details["threshold_series"] = line

    anomalies.sort(key=signals.anomaly_priority)

    # 6. Agent 判断层
    judgments = []
    judgment_rejections = []
    if do_llm and cfg_get(cfg, "agent.enabled", True) and anomalies:
        creds = get_llm_credentials(cfg.get("agent", {}))
        judge = EventJudge(creds["api_key"], creds["base_url"], creds["model"],
                           float(cfg.get("agent", {}).get("temperature", 0.1)),
                           int(cfg.get("agent", {}).get("max_tokens", 900)))
        if judge.enabled:
            max_events = int(cfg_get(cfg, "agent.max_events", 8))
            min_pp = int(cfg_get(cfg, "agent.min_per_platform", 2))
            per_ev = int(cfg_get(cfg, "agent.per_event_max", 1))
            # 事件级去重：同事件互斥合约（Fed 阶梯/Iran 候选人系列）只留代表，
            # 报告仍展示全部异动，仅 Agent 输入收窄
            agent_input = signals.dedupe_anomalies_by_event(anomalies, per_ev)
            judgments = judge.judge_batch(agent_input, max_events, min_per_platform=min_pp)
            judgment_rejections = judge.validation_failures

    # 7. 资产建议
    suggestions = signals.generate_asset_suggestions(markets, cfg)

    # 8. 落快照（供下次运行对比）
    store = SnapshotStore(project_path(cfg_get(cfg, "storage.sqlite_path", "data/snapshots.db")))
    try:
        saved = store.save_batch(markets, now_ts)
    finally:
        store.close()

    # 展示用 Top：各平台各自前 10（按成交额），保证小平台可见
    top_show: List[Market] = []
    for p in sorted({m.platform for m in markets}):
        top_show.extend([m for m in markets if m.platform == p][:10])

    payload = {
        "schema_version": 1,
        "status": "success" if markets else "no_data",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "platforms": platforms,
        "keywords": cfg_get(cfg, "runtime.keywords", []),
        "total_markets": len(markets),
        "snapshots_saved": saved,
        "top_markets": [report.market_view(m) for m in top_show],
        "threshold_series": ts_summaries,
        "anomalies": [a.to_dict() for a in anomalies],
        "cross_platform_pairs": [
            {
                "polymarket_question": a.question,
                "kalshi_question": b.question,
                "polymarket_prob": a.yes_pct,
                "kalshi_prob": b.yes_pct,
                "divergence_pts": round(abs(a.yes_probability - b.yes_probability) * 100, 1),
                "similarity": round(score, 3),
                "needs_review": True,
            }
            for a, b, score in cp_pairs
        ],
        "judgments": [j.to_dict() for j in judgments],
        "judgment_rejections": judgment_rejections,
        "asset_suggestions": suggestions,
        "suggestion_exclusions": [{"question": m.question, "reason": reason}
            for m in markets if (reason := signals.suggestion_block_reason(m))],
    }
    return payload


def save_outputs(payload, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    jp = os.path.join(output_dir, f"report_{tag}.json")
    mp = os.path.join(output_dir, f"report_{tag}.md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(report.build_markdown(payload))
    return jp, mp


def selftest() -> int:
    """离线自检：不联网验证分类/解析/信号/跨平台/格式核心逻辑。"""
    from src.models import Market, Anomaly
    from src.util import format_volume, days_until

    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {detail}"))
        if not cond:
            failures.append(name)

    print("\n=== 离线自检 ===\n")

    # 分类：词边界 + 主类型优先级 + 阈值型
    m = Market(market_id="1", platform="polymarket",
               question="Will the US impose tariffs on China?", yes_probability=0.7, no_probability=0.3)
    classify.annotate(m)
    check("词边界命中 tariff", "tariff" in m.event_types, str(m.event_types))
    check("子串误伤消除", "war" not in m.event_types)
    mt = Market(market_id="2", platform="polymarket", question="Will Bitcoin be above $150,000?",
                yes_probability=0.8, no_probability=0.2)
    classify.annotate(mt)
    check("阈值型判定 threshold", mt.question_shape == "threshold", mt.question_shape)

    # 有价值过滤：已过期/零流动性被剔除
    cfg = {"runtime": {"min_volume": 1000, "min_liquidity": 1}}
    good = Market(market_id="g", platform="polymarket", question="x", yes_probability=.5,
                  no_probability=.5, volume_total=5000, liquidity=100, days_to_resolution=10)
    expired = Market(market_id="e", platform="polymarket", question="x", yes_probability=.5,
                     no_probability=.5, volume_total=5000, liquidity=100, days_to_resolution=-5)
    dead = Market(market_id="d", platform="polymarket", question="x", yes_probability=.5,
                  no_probability=.5, volume_total=50000, liquidity=0, days_to_resolution=10)
    kept = signals.filter_valuable([good, expired, dead], cfg)
    check("过滤剔除已过期/零流动性", [k.market_id for k in kept] == ["g"], str([k.market_id for k in kept]))

    # 平台保底配额：量级差异下 Kalshi 不被全局截断吞掉
    pool = [Market(market_id=f"p{i}", platform="polymarket", question="x", yes_probability=.5,
                   no_probability=.5, volume_total=1_000_000 - i, liquidity=1, days_to_resolution=5)
            for i in range(20)] + [
           Market(market_id=f"k{i}", platform="kalshi", question="y", yes_probability=.5,
                  no_probability=.5, volume_total=50_000 - i, liquidity=1, days_to_resolution=5)
            for i in range(5)]
    trunc = signals.quota_truncate(pool, limit=10, min_per_platform=4)
    check("配额截断保留 Kalshi", len(trunc) <= 10 and sum(1 for t in trunc if t.platform == "kalshi") >= 4,
          str([(t.platform, t.market_id) for t in trunc]))
    check("无配额时全被 PM 吞掉（回归对照）",
          all(t.platform == "polymarket" for t in signals.quota_truncate(pool, 10, 0)))

    # 聪明钱信号：资金涌入
    inflow = Market(market_id="s", platform="polymarket", question="Will there be a ceasefire?",
                    yes_probability=0.5, no_probability=0.5, volume_24h=90000, volume_total=100000,
                    liquidity=5000, days_to_resolution=20)
    classify.annotate(inflow)
    sigs = signals.detect_smart_money([inflow], {"signals": {"volume_24h_share": 0.08}})
    check("识别资金涌入信号", bool(sigs) and "money_inflow" in sigs[0].details["signal_tags"], str(sigs))

    # 跨平台匹配 + 背离
    a = Market(market_id="a", platform="polymarket", question="Will the Fed cut interest rates in September?",
               yes_probability=0.75, no_probability=0.25)
    b = Market(market_id="b", platform="kalshi", question="Will the Fed cut interest rates in September?",
               yes_probability=0.55, no_probability=0.45, ticker="KX1")
    classify.annotate(a); classify.annotate(b)
    divs, pairs = cross_platform.match_markets([a, b], similarity_cutoff=0.6, divergence_pts=10)
    check("跨平台配对成功", len(pairs) == 1, str(len(pairs)))
    check("跨平台背离检出", len(divs) == 1 and divs[0].details["divergence_pts"] == 20.0, str(divs))

    # 高 embedding 相似度不能绕过命题门禁（mock 向量不触网）
    epm = Market(market_id="epm", platform="polymarket",
                 question="Will there be no change in Fed interest rates after the October meeting?",
                 yes_probability=.9, no_probability=.1)
    eks = Market(market_id="eks", platform="kalshi", ticker="KXFEDEMB",
                 question="Will the federal funds rate upper bound be above 3.75% after the Fed October meeting?",
                 yes_probability=.6, no_probability=.4)
    classify.annotate(epm); classify.annotate(eks)

    class _FakeEmb:
        enabled = True

        def embed(self, texts):
            vs = {epm.question: [1.0, 0.0], eks.question: [0.99, 0.14]}
            return {t: vs[t] for t in texts if t in vs}

    edivs, epairs = cross_platform.match_markets([epm, eks], embedder=_FakeEmb(),
                                                 embedding_cutoff=0.8, divergence_pts=10)
    check("拒绝同主题但不等价的跨平台命题", not epairs and not edivs)

    # 资产建议生成（新门禁需显式放宽 24h 门槛）
    cfg2 = {"runtime": {"suggest_top_n": 3, "suggest_min_volume_24h": 0}}
    sug = signals.generate_asset_suggestions([a], cfg2)
    check("资产建议生成", bool(sug) and sug[0]["event_type"] == "interest_rate", str(sug[:1]))

    # util
    check("format_volume 格式化", format_volume(1_500_000) == "$1.5M", format_volume(1_500_000))
    check("days_until 解析", isinstance(days_until("2030-01-01T00:00:00Z"), int))

    # 美国州级选举剔除（州级命中、全国性保留、地理名不误伤）
    check("剔除州级参议员盘", elections.is_us_state_election("Will Republicans win the Senate race in Texas?"))
    check("剔除州众议院盘", elections.is_us_state_election("Who will win the Texas House of Representatives?"))
    check("保留总统大选（全国性）", not elections.is_us_state_election("Will Trump win the 2028 presidential election?"))
    check("保留参议院控制权（全国性）", not elections.is_us_state_election("Who will control the US Senate after midterms?"))
    check("地理名 Georgia 不误伤", not elections.is_us_state_election("Will Russia invade Georgia?"))

    # 次国家级选举剔除（市长盘命中、央行人事不误伤）
    check("剔除台北市长盘", elections.is_local_election("Will Puma Shen win the next Taipei mayoral election?"))
    check("保留 Fed governor 人事盘", not elections.is_local_election("Will a Fed governor be replaced before 2027?"))
    check("保留总统大选", not elections.is_local_election("Will Trump win the 2028 presidential election?"))

    # 事件级去重 + 阈值系列隐含分布（Fed 阶梯）
    lad = [Market(market_id=f"KX{i}", platform="kalshi", event_key="KXFED-26DEC",
                  question=f"Will the upper bound of the federal funds rate be above {x}%?",
                  yes_probability=p, no_probability=1 - p, volume_total=v)
           for i, (x, p, v) in enumerate([(3.25, .985, 30_000), (3.75, .76, 40_000), (4.25, .205, 20_000)])]
    lanns = [Anomaly(market_id=m.market_id, platform=m.platform, question=m.question, anomaly_type="volume",
                     severity="high", current_probability=m.yes_probability,
                     details={"volume_total": m.volume_total}, market=m) for m in lad]
    ded = signals.dedupe_anomalies_by_event(lanns, 1)
    check("事件级去重后同事件只留 1 代表", len(ded) == 1 and ded[0].market_id == "KX1", str([d.market_id for d in ded]))
    tss = threshold_series.summarize(lad)
    check("阈值系列隐含分布（中位数插值）",
          bool(tss) and 3.75 < tss[0]["median"] < 4.25 and "隐含分布" in tss[0]["line"], str(tss[:1]))

    # 关键词相关性门禁 + 体育娱乐奖项剔除
    fedm = Market(market_id="f", platform="kalshi", question="Will the Fed cut interest rates in December?",
                  yes_probability=0.7, no_probability=0.3)
    baseb = Market(market_id="b", platform="kalshi",
                   question="Will more than 600 games be played in the 2027 Pro Baseball regular season?",
                   yes_probability=0.8, no_probability=0.2)
    bulg = Market(market_id="u", platform="kalshi",
                  question="Will Iliana Yotova win the 2026 Bulgarian presidential election?",
                  yes_probability=0.88, no_probability=0.12)
    for x in (fedm, baseb, bulg):
        classify.annotate(x)
    cfg_rel = {"runtime": {"keywords": ["Fed"], "require_keyword_match": True}}
    kept_rel = signals.filter_relevant([fedm, baseb, bulg], cfg_rel)
    check("相关性保留 Fed 盘", any(k.market_id == "f" for k in kept_rel), str([k.market_id for k in kept_rel]))
    check("剔除棒球赛程盘", not any(k.market_id == "b" for k in kept_rel))
    check("剔除无关小国选举盘", not any(k.market_id == "u" for k in kept_rel))
    check("保留体育描述不误伤 Nobel", not elections.is_off_theme("Will X win the Nobel Peace Prize?"))

    print(f"\n=== 自检结束: {'全部通过' if not failures else str(len(failures)) + ' 项失败'} ===\n")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prediction Market Agent")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--platforms", default=None, help="逗号分隔: polymarket,kalshi")
    parser.add_argument("--keywords", default=None, help="逗号分隔关键词")
    parser.add_argument("--min-volume", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-enrich", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    cfg = load_config(args.config)
    platforms = args.platforms.split(",") if args.platforms else None
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()] if args.keywords else None
    apply_overrides(cfg, platforms=platforms, keywords=keywords,
                    min_volume=args.min_volume, limit=args.limit, disable_llm=args.no_llm)

    logger.info("=" * 60)
    logger.info("开始流程: platforms=%s, keywords=%d 个",
                cfg_get(cfg, "runtime.platforms"), len(cfg_get(cfg, "runtime.keywords", [])))
    logger.info("=" * 60)

    payload = run_pipeline(cfg, do_enrich=not args.no_enrich, do_llm=not args.no_llm)

    if payload["status"] == "no_data":
        print("\n[结果] 未找到符合条件的市场")
        return 0

    report.print_summary(payload)
    output_dir = project_path(args.output_dir or cfg_get(cfg, "output.dir", "output"))
    jp, mp = save_outputs(payload, output_dir)
    print(f"\n[保存] {jp}\n[保存] {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
