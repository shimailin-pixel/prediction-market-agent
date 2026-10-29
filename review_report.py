"""离线复核已有 JSON 报告，不抓行情、不调用 LLM、不写快照。

只复核原报告已保存的候选，不宣称重新扫描原始全部市场。
用法：python review_report.py output/report_20261027_091918.json
"""
import argparse
import json
import re
from pathlib import Path

from src.models import Anomaly, Market
from src.analysis.classify import annotate
from src.analysis.signals import suggestion_block_reason, observation_action
from src.analysis.threshold_series import median_info, format_line
from src.agent.judge import Judgment
from src.agent.validation import validate_evidence
from src.report import build_markdown


def review(payload):
    payload = json.loads(json.dumps(payload))
    payload['review_note'] = ('原采集时点数据的离线复核；仅重新校验已保存的判断、建议和阈值阶梯，'
                              '未重新抓取全部市场或调用模型。')
    lookup = {}
    for row in payload.get('anomalies', []):
        m = annotate(Market(**row['market'])) if row.get('market') else None
        # 保留旧数据数值，将误导性标签说明改为可验证的统计口径。
        details = row['details']
        details['signal_reasons'] = [s.replace('资金正在涌入', '成交占比较高（不代表净资金流入）')
                                    .replace('存在巨鲸主导', '仅为可见头部样本集中度')
                                    for s in details.get('signal_reasons', [])]
        if m:
            details['prob_change_24h_pts'] = m.prob_change_24h_pts
            details['prob_change_7d_pts'] = m.prob_change_7d_pts
        a = Anomaly(**{**row, 'market': m})
        lookup[(a.market_id, a.anomaly_type)] = a
    for s in payload.get('threshold_series', []):
        info = median_info(s['ladder'], s['direction'])
        scope = ('期间最高值' if s['direction'] == 'above' else '期间最低值') if 'hit' in s['label'].lower() else '结算变量'
        comparator = ('≥' if s['direction'] == 'above' else '≤') if 'hit' in s['label'].lower() else ('>' if s['direction'] == 'above' else '<')
        s.update(info, scope=scope, comparator=comparator)
        s['line'] = scope + '：' + format_line(info['median'], s['ladder'], s['direction'], s['unit'], comparator)
    kept, rejected = [], []
    for j in payload.get('judgments', []):
        a = lookup.get((j['market_id'], j['anomaly_type']))
        reasons = []
        try:
            Judgment.model_validate(j)
        except ValueError:
            reasons.append('结构化字段不完整或数值越界')
        if a is None:
            reasons.append('缺少对应原始异动证据')
        else:
            # 旧 AI 文本依赖错误中位数时也必须拦截，不能仅改旁边的表格。
            if 'threshold_series' in a.details:
                a.details['threshold_series'] = '\n'.join(s['line'] for s in payload['threshold_series']
                    if a.market and s['platform'] == a.market.platform and s['event_key'] == a.market.event_key)
            reasons.extend(validate_evidence(j, a))
        if reasons:
            rejected.append({'market_id':j['market_id'], 'question':j['question'], 'reasons':reasons})
        else:
            j['validation_status'] = 'automated_screen_only'
            kept.append(j)
    payload['judgments'], payload['judgment_rejections'] = kept, rejected
    kept_suggestions, exclusions = [], []
    for s in payload.get('asset_suggestions', []):
        q = re.sub(r'\s*\(Yes [\d.]+%\)$', '', s['trigger'])
        m = annotate(Market(q, s['platform'], q, s['probability']/100, 1-s['probability']/100))
        reason = suggestion_block_reason(m)
        if reason:
            exclusions.append({'question':q, 'reason':reason})
        else:
            s['needs_review'] = True
            s['action'] = observation_action(m.primary_type, s['direction'])
            s['reason'] = '条件性传导假设，尚未核对资产现价与基准预期'
            kept_suggestions.append(s)
    payload['asset_suggestions'] = kept_suggestions
    payload['suggestion_exclusions'] = exclusions
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    result = review(json.loads(args.source.read_text(encoding='utf-8')))
    base = args.source.with_name(args.source.stem + '_reviewed')
    base.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    base.with_suffix('.md').write_text(build_markdown(result), encoding='utf-8')
    print(f"AI: {len(result['judgments'])} kept, {len(result['judgment_rejections'])} rejected; "
          f"suggestions: {len(result['asset_suggestions'])} kept, {len(result['suggestion_exclusions'])} withheld")
    print(base.with_suffix('.md'))


if __name__ == '__main__':
    main()
