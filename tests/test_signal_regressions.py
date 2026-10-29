"""离线回归：方向、命题一致性、合约量/换手率及旧数据库迁移。"""
import sqlite3

import pytest

from src.models import Market
from src.analysis.classify import annotate
from src.analysis.signals import generate_asset_suggestions
from src.analysis.cross_platform import match_markets
from src.analysis.tracker import track_changes
from src.data_source.kalshi import KalshiClient
from src.data_source.storage import SnapshotStore, _SCHEMA
from src.agent.prompt import build_evidence
from src.report import build_markdown


def market(question="Will the Fed hike rates in December 2026?", **kwargs):
    values = dict(market_id="m", platform="kalshi", ticker="KX",
                  question=question, yes_probability=.8, no_probability=.2,
                  volume_total=8000, volume_24h=1000,
                  volume_contracts=1200, open_interest_contracts=1000)
    values.update(kwargs)
    return annotate(Market(**values))


@pytest.mark.parametrize("verb,direction", [
    ("hike", "high"), ("increase", "high"), ("raise", "high"),
    ("cut", "low"), ("decrease", "low"), ("lower", "low"),
])
def test_rate_direction(verb, direction):
    result = generate_asset_suggestions([market(f"Will the Fed {verb} rates in December?")], {})
    assert result[0]["direction"] == direction


@pytest.mark.parametrize("question", [
    "Will the Fed hold rates in December?",
    "Will no Fed rate hikes happen in 2026?",
    "Will the Fed cut or hike rates in December?",
    "Will the Fed chair resign in December?",
])
def test_ambiguous_rate_propositions_do_not_imply_cuts(question):
    assert generate_asset_suggestions([market(question)], {}) == []


class HighSimilarity:
    enabled = True

    def embed(self, texts):
        return {t: [1., 0.] for t in texts}


@pytest.mark.parametrize("left,right", [
    ("Will the Fed cut rates in September 2026?", "Will the Fed cut rates in September 2027?"),
    ("Will oil be above $80 in December?", "Will oil be above $90 in December?"),
    ("Will oil be above $80 in December?", "Will oil be below $80 in December?"),
    ("Will oil be above $80 in December?", "Will oil be at least $80 in December?"),
    ("Will the Fed cut rates by 25 bps?", "Will the Fed cut rates by 50 bps?"),
    ("Will the Fed cut rates in September?", "Will the Fed hike rates in September?"),
    ("Will the Fed cut rates before September?", "Will the Fed cut rates after September?"),
    ("Will the Fed cut rates in September?", "Will the Fed not cut rates in September?"),
    ("Will Russia invade Ukraine?", "Will Ukraine invade Russia?"),
    ("Will inflation be above -1% in December?", "Will inflation be above 1% in December?"),
    ("Will oil be above $80 in December?", "Will oil be above 80% in December?"),
])
@pytest.mark.parametrize("embedder", [None, HighSimilarity()])
def test_different_propositions_rejected(left, right, embedder):
    pm = market(left, platform="polymarket")
    ks = market(right, yes_probability=.2)
    assert match_markets([pm, ks], embedder=embedder) == ([], [])


@pytest.mark.parametrize("embedder", [None, HighSimilarity()])
def test_equivalent_wording_still_matches(embedder):
    pm = market("Will the Federal Reserve decrease interest rates in December 2026?", platform="polymarket")
    ks = market("Will the Fed cut rates in December 2026?", yes_probability=.2)
    anomalies, pairs = match_markets([pm, ks], embedder=embedder)
    assert len(pairs) == len(anomalies) == 1
    assert anomalies[0].details["needs_review"]


def test_same_title_different_end_dates_rejected():
    pm = market(platform="polymarket", end_date="2026-12-01T00:00:00Z")
    ks = market(yes_probability=.2, end_date="2027-12-01T00:00:00Z")
    assert match_markets([pm, ks], embedder=HighSimilarity()) == ([], [])


class History:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else [
            dict(ts=t, yes_probability=.5, volume_total=v * .5, volume_contracts=v)
            for t, v in [(100, 1000), (200, 1010), (300, 1020)]
        ]

    def history(self, _):
        return self.rows


def spikes(m, store=None, now=400, threshold=.10):
    return [a for a in track_changes([m], store or History(),
            {"signals": {"kalshi_min_interval_turnover": threshold}}, now)
            if a.anomaly_type == "volume_spike"]


def test_price_change_without_trades_never_triggers_volume_spike():
    assert spikes(market(volume_contracts=1020, volume_total=816)) == []


@pytest.mark.parametrize("oi", [None, 0, -1, float("nan"), float("inf")])
def test_unavailable_open_interest_does_not_pass_turnover_gate(oi):
    assert spikes(market(open_interest_contracts=oi)) == []


def test_turnover_boundary_and_config_override():
    m = market(volume_contracts=1120, open_interest_contracts=1000)
    assert spikes(m, threshold=.101) == []
    hit = spikes(m, threshold=.10)[0]
    assert hit.details["interval_turnover"] == .10
    assert hit.details["increment_contracts"] == 100
    assert hit.details["multiple"] == 10
    assert "increment_usd" not in hit.details
    assert "10.0%" in build_evidence(hit.details, hit.anomaly_type)
    assert "10.0%" in build_markdown({"anomalies": [hit.to_dict()]})


def test_turnover_alone_does_not_bypass_volume_multiple():
    assert spikes(market(volume_contracts=1040, open_interest_contracts=100)) == []


def test_longer_sampling_interval_is_not_a_volume_spike():
    # 历史每秒 .1 张；当前隔 1000 秒新增 100 张，速率没变。
    assert spikes(market(volume_contracts=1120), now=1300) == []


def test_legacy_kalshi_history_does_not_fall_back_to_revalued_usd():
    rows = History().rows
    for row in rows:
        row.pop("volume_contracts")
    assert spikes(market(), History(rows)) == []


def test_contract_counter_reset_does_not_trigger_volume_spike():
    assert spikes(market(volume_contracts=50)) == []


def test_polymarket_dollar_volume_tracking_still_works():
    hit = spikes(market(platform="polymarket", volume_total=700))[0]
    assert hit.details["volume_unit"] == "usd"
    assert hit.details["increment_usd"] == 190


def test_kalshi_parser_preserves_contracts_independent_of_price():
    client = KalshiClient({}, http=object())
    raw = dict(ticker="KX", title="Fed rate cut?", volume_fp="10000.00",
               open_interest_fp="500.00", last_price_dollars="0.50")
    old = client._to_market(raw, set())
    raw["last_price_dollars"] = "0.80"
    new = client._to_market(raw, set())
    assert old.volume_contracts == new.volume_contracts == 10000
    assert old.open_interest_contracts == new.open_interest_contracts == 500
    assert old.volume_total != new.volume_total


def test_legacy_database_migration_and_round_trip(tmp_path):
    path = str(tmp_path / "snapshots.db")
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)
        conn.execute("INSERT INTO snapshots VALUES ('m','kalshi',1,.5,100,10,20)")
    store = SnapshotStore(path)
    try:
        assert store.history("m")[0]["volume_contracts"] is None
        store.save_snapshot(market(), 2)
        store.save_batch([market(volume_contracts=1300)], 3)
        assert store.history("m")[-1]["volume_contracts"] == 1300
        assert store.history("m")[-1]["open_interest_contracts"] == 1000
        assert store.history("m", limit=1)[0]["ts"] == 3
    finally:
        store.close()

    # 幂等升级，原始历史不被覆盖。
    store = SnapshotStore(path)
    try:
        assert len(store.history("m")) == 3
    finally:
        store.close()


def test_pipeline_persists_contracts_and_reports_turnover(tmp_path, monkeypatch):
    import main
    from datetime import datetime

    clock = [100]

    class Clock:
        @staticmethod
        def now():
            return datetime.fromtimestamp(1_700_000_000 + clock[0])

    monkeypatch.setattr(main, "datetime", Clock)
    cfg = {"runtime": {"platforms": ["kalshi"], "keywords": ["Fed"]},
           "signals": {"use_embedding": False},
           "storage": {"sqlite_path": str(tmp_path / "pipeline.db")}}
    for timestamp, count in [(100, 1000), (200, 1010), (300, 1020), (400, 1120)]:
        clock[0] = timestamp
        m = market(volume_contracts=count, liquidity=100)
        monkeypatch.setattr(main, "fetch_all", lambda cfg: ([m], ["kalshi"]))
        payload = main.run_pipeline(cfg, do_enrich=False, do_llm=False)
        assert payload["snapshots_saved"] == 1
        hits = [a for a in payload["anomalies"] if a["anomaly_type"] == "volume_spike"]
        if timestamp < 400:
            assert hits == []
    assert len(hits) == 1
    assert hits[0]["details"]["interval_turnover"] == .1
    assert hits[0]["market"]["volume_contracts"] == 1120
    assert "10.0%" in main.report.build_markdown(payload)
