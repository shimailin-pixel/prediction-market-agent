import json
from types import SimpleNamespace

import pytest

from src.models import Market, Anomaly
from src.analysis.classify import annotate
from src.analysis.signals import generate_asset_suggestions, dedupe_anomalies_by_event
from src.analysis.threshold_series import median_info, summarize
from src.agent.judge import EventJudge
from src.report import build_markdown


def market(q, p=.8):
    return annotate(Market('m', 'kalshi', q, p, 1-p, volume_24h=1000, volume_total=10000))


@pytest.mark.parametrize('q,p', [
    ('Fed emergency rate cut before 2027?', .026),
    ('Will no Fed rate cuts happen in 2026?', .96),
    ('Will the Fed Hike rates by 0bps?', .8),
    ('Will the Fed cut rates by 0 bps?', .8),
    ('Will oil be below $85 and the Republicans win the Senate?', .8),
    ('Will WTI oil hit (HIGH) $115 in October?', .029),
    ('US-Iran Final Nuclear Deal by December 31, 2026?', .125),
    ('US-Iran Final Nuclear Deal by December 31, 2026?', .9),
    ('Will Reza Pahlavi lead Iran in 2026?', .9),
])
def test_report_misread_propositions_no_longer_generate_trades(q, p):
    assert generate_asset_suggestions([market(q, p)], {}) == []


def test_nuclear_agreement_has_distinct_classification():
    assert market('Will Iran sign a nuclear agreement?').primary_type == 'nuclear_agreement'
    assert market('Will Iran use a nuclear weapon?').primary_type == 'nuclear'


@pytest.mark.parametrize('direction,ladder,status,bound', [
    ('above', [(2, .99), (3, .95), (4, .84)], 'above_range', 4),
    ('above', [(105, .435), (110, .115), (115, .029)], 'below_range', 105),
    ('below', [(2, .1), (3, .2), (4, .3)], 'above_range', 4),
    ('below', [(2, .7), (3, .8), (4, .9)], 'below_range', 2),
    ('above', [(2, .9), (3, .3), (4, .8)], 'invalid', None),
    ('above', [(2, .9), (2, .8), (4, .1)], 'invalid', None),
])
def test_median_boundaries(direction, ladder, status, bound):
    info = median_info(ladder, direction)
    assert info['median'] is None
    assert info['median_status'] == status
    assert info['median_bound'] == bound


def test_threshold_output_explains_bound_and_strict_comparison():
    ms = [market(f'Will the Fed rate be above {x}%?', p) for x, p in [(2,.99),(3,.95),(4,.84)]]
    for m in ms:
        m.event_key = 'fed'
    s = summarize(ms)[0]
    assert s['median'] is None
    assert '高于 4%' in s['line']
    assert 'P(>4%)' in s['line']


def response(**changes):
    data = dict(event_summary='当前概率为80.0%，缺少变化证据。', event_category='Monetary Policy',
                category_cn='货币政策', impact_on_a_share='中性',
                impact_direction={k: '中性：证据不足' for k in ['A股','美元','大宗商品','黄金']},
                confidence=.5, trade_bias='观望，等待传导确认', rationale='当前概率80.0%，不能确定资产方向。',
                risk_note='结算规则和后续数据尚需核对')
    data.update(changes)
    return data


def judge_with(data):
    judge = EventJudge('', '', '')
    judge.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(data, ensure_ascii=False)))]))))
    a = Anomaly('m', 'kalshi', 'Will the Fed hike rates?', 'volume', 'medium', .8,
                {'volume_total': 10000}, market('Will the Fed hike rates?'))
    return judge.judge(a), judge.validation_failures


@pytest.mark.parametrize('changes', [
    {'impact_direction': {'A股':'中性','美元':'中性','大宗商品':'中性','黄金':''}},
    {'impact_direction': {'A股':'中性','美元':'中性','大宗商品':'中性'}},
    {'confidence': 1.2}, {'confidence': float('nan')},
    {'rationale': '历史校准度高，结算准确率大于85%。'},
    {'event_summary': '概率升至80.0%，预期强化。'},
    {'rationale': '资金加速流入，巨鲸涌入。'},
])
def test_invalid_ai_output_is_rejected_with_reason(changes):
    result, failures = judge_with(response(**changes))
    assert result is None
    assert failures and failures[0]['reasons']


def test_supported_ai_output_survives():
    result, failures = judge_with(response())
    assert result and not failures
    assert result.validation_status == 'automated_screen_only'


def test_dynamic_event_wins_dedupe_and_report_preserves_threshold():
    m = market("Will the upper bound of the federal funds rate be above 3.75% following the Fed's Dec 9, 2026 meeting?")
    m.event_key = 'fed'
    static = Anomaly('m','kalshi',m.question,'volume+whale','high',.8, {'volume_total':10000},m)
    dynamic = Anomaly('m','kalshi',m.question,'prob_spike','high',.8,
                      {'change_pts':20, 'old_probability':.6, 'new_probability':.8},m)
    assert dedupe_anomalies_by_event([static,dynamic]) == [dynamic]
    md = build_markdown({'anomalies':[static.to_dict(),dynamic.to_dict()]})
    assert md.index('prob_spike') < md.index('volume+whale')
    assert m.question in md and '3.75%' in md and 'Dec 9, 2026' in md


def test_numeric_evidence_supports_units_months_and_signed_change():
    from src.agent.validation import number_tokens, validate_evidence
    assert number_tokens('储备2.8亿桶，成交3.7M美元') == number_tokens('280M 370万')
    m = market('Will US crude oil reserves fall to 280M by October 30, 2026?', .085)
    a = Anomaly('m', 'kalshi', m.question, 'prob_spike', 'high', .085,
                {'change_pts': -21.5, 'old_probability': .3, 'new_probability': .085}, m)
    assert validate_evidence({'event_summary':'10月30日储备2.8亿桶事件概率下降21.5点至8.5%。'}, a) == []
    assert validate_evidence({'rationale':'历史准确率85%。'}, a)


def test_offline_review_preserves_source_and_removes_unsupported_output():
    from copy import deepcopy
    from review_report import review
    m = market('Will the Fed hike rates?', .8)
    a = Anomaly('m','kalshi',m.question,'volume','medium',.8,{},m)
    j = dict(market_id='m', platform='kalshi', question=m.question,
             anomaly_type='volume', current_probability=.8, timestamp='2026-10-27',
             **response(rationale='历史准确率85%。'))
    source = {'anomalies':[a.to_dict()], 'judgments':[j],
              'asset_suggestions':[{'trigger':'Fed emergency rate cut before 2027? (Yes 2.6%)',
                'platform':'polymarket','probability':2.6}],
              'threshold_series':[{'platform':'kalshi','event_key':'fed','label':'Fed rate above',
                  'direction':'above','unit':'%','ladder':[(2,.99),(3,.95),(4,.84)],'median':4}]}
    original = deepcopy(source)
    result = review(source)
    assert source == original
    assert result['judgments'] == [] and result['judgment_rejections']
    assert result['asset_suggestions'] == [] and result['suggestion_exclusions']
    assert result['threshold_series'][0]['median'] is None
    assert '高于 4%' in result['threshold_series'][0]['line']
    assert '离线复核' in build_markdown(result)
