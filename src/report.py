"""输出层：把 payload 渲染为 Markdown 报告与终端摘要。"""

from __future__ import annotations

from typing import Any, Dict, List

from .util import format_volume, short_question
from .agent.prompt import build_evidence
from .analysis.signals import anomaly_priority
from .models import Anomaly


def _v(val: Any, fallback: str = "—") -> str:
    """展示值兜底：键存在但为空（LLM 漏填）时也给占位符。"""
    s = str(val or "").strip()
    return s or fallback


def full_question(question, url=""):
    text = question.replace("|", "\\|").replace("\n", " ").replace("[", "\\[").replace("]", "\\]")
    if url.startswith("https://") and not any(c in url for c in "() "):
        return f"[{text}]({url})"
    return text


def build_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# 预测市场监控报告")
    lines.append("")
    lines.append(f"- 生成时间：{payload.get('timestamp')}")
    lines.append(f"- 数据源：{', '.join(payload.get('platforms', []))}")
    lines.append(f"- 关键词：{', '.join(payload.get('keywords', []))}")
    lines.append(f"- 有价值市场：{payload.get('total_markets')} | "
                 f"异动信号：{len(payload.get('anomalies', []))} | "
                 f"AI 判断：{len(payload.get('judgments', []))} | "
                 f"资产建议：{len(payload.get('asset_suggestions', []))}")
    lines.append("")

    if payload.get("review_note"):
        lines.append(f"- 数据范围：{payload['review_note']}")
    lines.append("- 解读限制：AI 仅通过自动规则筛查，模型置信度未经校准；未验证资产收益或当前交易机会。")
    lines.append("- 口径：Kalshi 美元金额为合约数×当前概率的展示估计，未平仓量不等于挂单深度。")
    lines.append("")
    anom = payload.get("anomalies", [])
    def rank(row):
        return anomaly_priority(Anomaly(**{k: v for k, v in row.items() if k != "market"}))
    ordered = sorted(anom, key=rank)
    dynamic = [a for a in ordered if rank(a)[0] < 3]
    static = [a for a in ordered if rank(a)[0] == 3]
    for title, rows, limit in (("动态变化与跨平台候选", dynamic, len(dynamic)),
                               ("静态活跃度与持仓特征（不代表新变化）", static, 10)):
        if not rows:
            continue
        lines.extend([f"## {title}（{len(rows)} 条）", ""])
        for a in rows[:limit]:
            lines.append(f"- [{a['severity']}] {a['anomaly_type']} | {a['platform']} | {full_question(a['question'])}")
            lines.append(build_evidence(a.get("details", {}), a["anomaly_type"], include_limits=False))
        if len(rows) > limit:
            lines.append(f"- 其余 {len(rows)-limit} 条静态特征见同名 JSON。")
        lines.append("")

    top = payload.get("top_markets", [])
    if top:
        # 分平台展示：避免小平台（Kalshi）在全局成交额排序里完全隐身
        by_p: Dict[str, List[Dict[str, Any]]] = {}
        for m in top:
            by_p.setdefault(m["platform"], []).append(m)
        for p, ms in sorted(by_p.items()):
            lines.append(f"## 成交额 Top 市场 · {p}（前 {min(len(ms), 10)}）")
            lines.append("")
            lines.append("| # | 问题 | Yes (%) | 累计成交 | 24h成交 | 类型 | 距结算 |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for i, m in enumerate(ms[:10], 1):
                days = m.get("days_to_resolution")
                # 智能截断：Kalshi 阶梯盘区分信息（阈值/会议）在尾部，定长截头会重复
                lines.append(
                    f"| {i} | {full_question(m['question'], m.get('url', ''))} | {m['yes_pct']} | "
                    f"{m['volume_total_display']} | {m['volume_24h_display']} | "
                    f"{'/'.join(m.get('event_types', [])) or '-'} | "
                    f"{(str(days)+'天' if days is not None else '-')} |"
                )
            lines.append("")

    ts = payload.get("threshold_series", [])
    if ts:
        lines.append("## 阈值系列隐含分布（同事件多合约聚合）")
        lines.append("")
        for s in ts[:10]:
            lines.append(f"- **[{s['platform']}] {s['label']}**：{s['line']}")
        lines.append("")

    cp = payload.get("cross_platform_pairs", [])
    if cp:
        lines.append("## 跨平台候选对照（结算规则需复核）")
        lines.append("")
        lines.append("| Polymarket | Kalshi | PM概率 | KS概率 | 背离(点) | 相似度 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for pair in cp[:15]:
            lines.append(
                f"| {full_question(pair['polymarket_question'])} | {full_question(pair['kalshi_question'])} | "
                f"{pair['polymarket_prob']}% | {pair['kalshi_prob']}% | "
                f"{pair['divergence_pts']} | {pair['similarity']} |"
            )
        lines.append("")

    juds = payload.get("judgments", [])
    if juds:
        lines.append("## Agent 判断层（结构化解读）")
        lines.append("")
        for j in juds:
            lines.append(f"### [{j['platform']}] {full_question(j['question'])}")
            lines.append("")
            lines.append(f"- 异动类型：{j['anomaly_type']} | 当前概率：{round(j['current_probability']*100,1)}% "
                         f"| 类别：{j['category_cn'] or j['event_category']} | 模型自评（未经校准）：{round(j['confidence']*100)}%")
            lines.append(f"- 摘要：{j['event_summary']}")
            idir = j.get("impact_direction", {})
            lines.append(f"- 影响：A股={_v(idir.get('A股'))} | 美元={_v(idir.get('美元'))} | "
                         f"大宗={_v(idir.get('大宗商品'))} | 黄金={_v(idir.get('黄金'))}")
            lines.append(f"- 交易倾向：{j['trade_bias']}")
            lines.append(f"- 依据：{j['rationale']}")
            if j.get("risk_note"):
                lines.append(f"- 风险：{j['risk_note']}")
            lines.append("")

    rejected = payload.get("judgment_rejections", [])
    if rejected:
        lines.extend(["## 未通过校验的 AI 解读", ""])
        for item in rejected:
            lines.append(f"- {full_question(item['question'])}：{'；'.join(item['reasons'])}")
        lines.append("")

    sug = payload.get("asset_suggestions", [])
    if sug:
        lines.append("## 条件性资产观察（需复核）")
        lines.append("")
        lines.append("| 事件类型 | 方向 | 平台 | 操作 | 触发 | 距50%偏离（点，非置信度） | 需复核 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for s in sug:
            lines.append(
                f"| {s['event_type']} | {s['direction']} | {s['platform']} | {s['action']} | "
                f"{full_question(s['trigger'])} | {s['conviction']} | {'是' if s.get('needs_review') else '否'} |"
            )
        lines.append("")

    exclusions = payload.get("suggestion_exclusions", [])
    if exclusions:
        lines.extend([f"## 暂不生成方向建议（{len(exclusions)} 个候选）", ""])
        for item in exclusions[:10]:
            lines.append(f"- {full_question(item['question'])}：{item['reason']}")
        if len(exclusions) > 10:
            lines.append("- 其余候选及原因见同名 JSON。")
        lines.append("")

    lines.append("## 风险提示")
    lines.append("")
    lines.append("- 预测市场概率反映参与者集体预期，不等于事件必然发生。")
    lines.append("- 资产建议由关键词规则 + Agent 定性判断生成，落地前需人工核对语义与标的映射。")
    lines.append("- 阈值型问题（如 above $110）的概率含义与事件型不同，标记为需复核。")
    lines.append("- 本报告不构成投资建议。")
    lines.append("")
    return "\n".join(lines)


def print_summary(payload: Dict[str, Any]) -> None:
    print("=" * 70)
    print("  Prediction Market Agent — 分析摘要")
    print(f"  数据源: {', '.join(payload.get('platforms', []))} | 关键词: {len(payload.get('keywords', []))} 个")
    print(f"  有价值市场: {payload.get('total_markets')} | 异动: {len(payload.get('anomalies', []))} | "
          f"AI判断: {len(payload.get('judgments', []))} | 建议: {len(payload.get('asset_suggestions', []))}")
    print("=" * 70)

    top = payload.get("top_markets", [])
    if top:
        by_p: Dict[str, list] = {}
        for m in top:
            by_p.setdefault(m["platform"], []).append(m)
        for p, ms in sorted(by_p.items()):
            print(f"\n[{p} 成交额 Top 5]")
            for i, m in enumerate(ms[:5], 1):
                print(f"  {i}. {short_question(m['question'], 68)}")
                print(f"     Yes {m['yes_pct']}% | 累计 {m['volume_total_display']} | 24h {m['volume_24h_display']} "
                      f"| {'/'.join(m.get('event_types', [])) or '-'}")

    ts = payload.get("threshold_series", [])
    if ts:
        print(f"\n[阈值系列隐含分布] {len(ts)} 组")
        for s in ts[:4]:
            print(f"  - [{s['platform']}] {s['label'][:78]}: {s['line']}")

    juds = payload.get("judgments", [])
    if juds:
        print(f"\n[Agent 判断] {len(juds)} 条")
        for j in juds:
            print(f"  - [{j['platform']}/{j['anomaly_type']}] {short_question(j['question'], 56)}")
            print(f"     概率 {round(j['current_probability']*100,1)}% | 置信 {round(j['confidence']*100)}% "
                  f"| {j['category_cn'] or j['event_category']}")
            print(f"     倾向: {j['trade_bias']}")

    cp = payload.get("cross_platform_pairs", [])
    if cp:
        print(f"\n[跨平台同事件] {len(cp)} 对")
        for pair in cp[:6]:
            print(f"  - PM {pair['polymarket_prob']}% vs KS {pair['kalshi_prob']}% "
                  f"(背离 {pair['divergence_pts']}pt, 相似 {pair['similarity']})")
            print(f"    {short_question(pair['polymarket_question'], 60)}")

    sug = payload.get("asset_suggestions", [])
    if sug:
        print(f"\n[资产建议] {len(sug)} 条")
        for s in sug[:10]:
            flag = " (需复核)" if s.get("needs_review") else ""
            print(f"  - [{s['event_type']}/{s['direction']}] {s['action']}{flag}")
            print(f"     触发: {s['trigger'][:64]}")


def market_view(m, as_dict: bool = True) -> Dict[str, Any]:
    """Market → 输出视图（补展示用格式化字段）。"""
    d = m.to_dict()
    d["yes_pct"] = m.yes_pct
    d["volume_total_display"] = format_volume(m.volume_total)
    d["volume_24h_display"] = format_volume(m.volume_24h)
    return d
