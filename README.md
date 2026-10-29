# Prediction Market Agent

> 预测市场数据源 + 分析框架 + Agent 判断层
>
> 从 **Polymarket** 与 **Kalshi** 的公开 API 获取实时预测市场数据，提取有价值的市场，
> 追踪概率变化、检测异常信号，并参考 [TradingAgents](https://github.com/TauricResearch/TradingAgents)
> 的思路，用一个 **Agent 判断层**（prompt + 结构化输出）把海外事件概率变化翻译成对 A 股等大类资产的方向性判断。

预测市场是「用真金白银投票」的概率引擎。把两个主流平台的实时定价、成交动态与跨平台背离，
加工成可解读的信号，是资产配置的一条前瞻性线索。

---

## ✨ 特性

- **双平台数据接入**：Polymarket（Gamma + CLOB + Data）与 Kalshi（`/markets` + 成交时间窗口），字段差异在数据层归一化。
- **分析框架**：有价值市场过滤 → 事件分类 → 聪明钱信号 → 概率/成交量动态异动 → 跨平台定价背离。
- **embedding 语义匹配**：跨平台同事件配对用 DashScope 文本向量余弦相似度（失败自动降级回文本相似度）。
- **事件级去重**：同事件互斥合约（Fed 利率阶梯、伊朗候选人系列）只留代表进 Agent，报告仍展示全部异动。
- **阈值系列隐含分布**：同事件多个阈值合约折成一行「隐含中位数 + 两端概率」，并作为结构化证据送进 Agent。
- **多层噪音门禁**：平台保底配额、分平台放量基准、次国家级选举/噱头盘/尾概率彩票剔除。
- **Agent 判断层**：OpenAI 兼容端点（默认通义千问），多视角推理 + pydantic 结构化输出；事实字段由代码填充，模型不臆测数字。
- **概率追踪**：SQLite 落历史快照，跨运行对比检测概率跳变与成交量脉冲。
- **优雅降级**：无 LLM 密钥仍跑通「数据 + 信号」；httpx 缺失自动回落 urllib；单接口失败不影响主流程。
- **双输出**：机器可读 JSON + 人读 Markdown 报告。

---

## 🚀 快速开始

```bash
# 1. 安装依赖
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. 配置（可选：Agent 判断层需要 LLM 密钥；不配也能跑数据+信号）
cp .env.example .env    # 填入 DASHSCOPE_API_KEY 或 OPENAI_API_KEY

# 3. 运行
python main.py                         # 默认双平台 + 内置关键词
python main.py --selftest              # 离线自检（不联网，验证核心逻辑）
python main.py --keywords "Iran,Fed,tariff"
python main.py --platforms polymarket  # 只跑一个平台
python main.py --no-llm                # 跳过 Agent 判断
python main.py --no-enrich             # 跳过 CLOB/Data 增强（更快）
python main.py --min-volume 50000 --limit 200
```

结果写入 `output/report_<时间戳>.json` 与 `.md`。

> **网络说明**：市场数据读取（公开端点）无需任何 key；国内访问 Polymarket/Kalshi 通常需 HTTP 代理，
> 在 `.env` 设 `HTTPS_PROXY` 即可。所有阈值集中在 `config.yaml`，命令行可覆盖。

---

## 🏗️ 框架设计

### 流水线

```
                    ┌─────────────────────────────────────────────┐
  关键词 ──────────▶│  数据接入层  data_source/                     │
                    │   • PolymarketClient  (Gamma/CLOB/Data)      │
                    │   • KalshiClient      (/markets + 时间窗口)   │
                    └───────────────┬─────────────────────────────┘
                                    │  归一化为统一 Market
                                    ▼
                    ┌─────────────────────────────────────────────┐
                    │  分析框架  analysis/                          │
                    │   1. classify    事件类型 + 问题形态           │
                    │   2. elections   州级/次国家级选举噪音剔除     │
                    │   3. signals     有价值过滤 + 聪明钱信号 + 建议 │
                    │   4. tracker     概率/成交量动态异动（需历史） │
                    │   5. cross_platform  跨平台匹配(embedding)背离  │
                    │   6. threshold_series  阈值阶梯→隐含分布       │
                    └───────────────┬─────────────────────────────┘
                                    │  Anomaly（异动）列表
                                    ▼
                    ┌─────────────────────────────────────────────┐
                    │  Agent 判断层  agent/                         │
                    │   prompt(多视角) + LLM + pydantic 结构化输出  │
                    │   → 对 A股/美元/大宗商品/黄金 的方向判断      │
                    └───────────────┬─────────────────────────────┘
                                    ▼
                    输出层 report/  ──▶  JSON + Markdown
                    存储 storage    ──▶  SQLite 快照（供下次对比）
```

### 目录结构

```
.
├── main.py                     # CLI 入口 + 流水线编排 + 离线自检
├── config.yaml                 # 全部阈值 / 端点 / 关键词
├── requirements.txt
├── .env.example                # LLM 密钥 / 代理样例
├── src/
│   ├── http.py                 # HttpClient：httpx 优先，urllib 降级，重试/退避/代理
│   ├── models.py               # 统一 Market / Anomaly 数据模型（跨平台单一货币）
│   ├── config.py               # YAML + .env 配置加载
│   ├── util.py                 # 安全转换 / 日期 / 金额格式化
│   ├── report.py               # Markdown 报告 + 终端摘要
│   ├── data_source/
│   │   ├── polymarket.py       # Gamma 检索 + CLOB 概率历史 + Data 持仓集中度
│   │   ├── kalshi.py           # /markets + 成交时间窗口（2026 字段迁移适配）
│   │   └── storage.py          # SQLite 快照存储
│   ├── analysis/
│   │   ├── classify.py         # 事件类型 / 问题形态 / 资产映射
│   │   ├── signals.py          # 有价值过滤 + 聪明钱信号 + 资产建议
│   │   ├── tracker.py          # 概率跳变 / 成交量脉冲（对比历史快照）
│   │   └── cross_platform.py   # 跨平台匹配 + 背离检测
│   └── agent/
│       ├── prompt.py           # SYSTEM + JUDGE prompt + 证据拼装
│       └── judge.py            # EventJudge：LLM 调用 + pydantic 校验
├── examples/sample_output.json # 一次真实运行的样例输出
└── tests/                      # pytest 单元测试
```

---

## 🗂️ 数据结构

### `Market`（跨平台统一市场）

数据接入层把两个平台差异极大的原始字段，归一化为同一套语义（`src/models.py`）。金额一律 **USD**，概率一律 **0~1**。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `market_id` | str | 平台内唯一 ID（PM market id / KS ticker） |
| `platform` | str | `polymarket` \| `kalshi` |
| `question` | str | 市场问题文本 |
| `yes_probability` / `no_probability` | float | YES/NO 概率（0~1） |
| `volume_24h` / `volume_total` | float | 24h / 累计成交额（USD）；Kalshi 用「合约数 × 概率」估计 |
| `liquidity` | float | 挂单流动性（USD）；Kalshi 用未平仓合约美元代理 |
| `event_types` | list[str] | 命中的事件类型（可多值） |
| `primary_type` | str | 主类型（按优先级取最具体） |
| `question_shape` | str | `threshold`（价格水平）\| `event`（会不会发生） |
| `end_date` / `days_to_resolution` | str / int | 结算时间 / 距结算天数（负=已过期） |
| `event_key` | str | 事件归组键（PM event slug / KS event_ticker），去重/建议/阈值系列皆基于它 |
| `condition_id`/`token_id`/`ticker`/`category`/`url` | | 平台专属句柄 |
| `volume_24h_share` | float? | 24h 成交 / 累计（资金涌入度） |
| `prob_change_24h_pts` / `prob_change_7d_pts` | float? | 概率动量（增强层 CLOB 历史，百分点） |
| `concentration` | dict? | 持仓集中度（Polymarket Data API） |

> Kalshi 美元口径说明：1 张合约到期支付 $1，故 `成交额(USD) ≈ 成交合约数 × YES 概率`；
> 2026 Q1 起整数字段迁移为字符串定点字段（`volume_fp` / `yes_bid_dollars` 等），数据层已适配。

### `Anomaly`（异动 / 信号）

| 字段 | 说明 |
| --- | --- |
| `anomaly_type` | `prob_spike` / `volume_spike` / `cross_platform_divergence` / 聪明钱组合标签（`money_inflow+volume+...`） |
| `severity` | `high` \| `medium` |
| `current_probability` | 当前 YES 概率 |
| `details` | 证据明细（跳变点数、成交倍数、背离概率、信号原因等） |
| `market` | 附带完整 Market 上下文 |

### Agent 判断输出 `Judgment`（pydantic 结构化）

事实字段（`market_id`/`platform`/`anomaly_type`/`current_probability`/`timestamp`）由代码填充，
分析字段由 LLM 生成：`event_summary`、`event_category`、`category_cn`、`impact_on_a_share`、
`impact_direction{A股,美元,大宗商品,黄金}`、`confidence(0~1)`、`trade_bias`、`rationale`、`risk_note`。

---

## 🧠 信号与异动检测

分析框架分**静态聪明钱**与**动态异动**两类：

**聪明钱信号**（`signals.detect_smart_money`，当前快照即可判定）
- `money_inflow`：24h 成交 / 累计成交 ≥ 阈值 → 资金正在涌入；
- `volume`：累计成交额远超**本平台**均值（分平台基准，防止 Kalshi 被 Polymarket 全场均值淹没）；
- `prob_momentum` / `prob_trend`：24h / 7d 概率显著变动（来自 CLOB 概率历史）；
- `whale`：最大单一钱包占可见头部持仓比例过高（来自 Data `/holders`）。
  ≥2 类命中记为 `high`。

**动态异动**（`tracker.track_changes`，需跨运行历史快照）
- `prob_spike`：当前概率相对上一快照跳变 ≥ 阈值；
- `volume_spike`：当前成交速率 ≥ 历史有效区间平均速率 × `signals.volume_multiple`，按实际区间秒数归一化。
  - Kalshi 使用原始 `volume_fp` 合约数的增量，绝不对「累计合约数 × 当前概率」求差；美元估计仅用于原有展示/过滤。
  - 同时要求 **区间换手率 = 区间成交合约增量 / 当前未平仓合约数** ≥ `signals.kalshi_min_interval_turnover`，默认 `0.10`（10%）。这是区间换手率，不是 24h 或年化换手率。
  - 缺失/非正未平仓量、缺失原始合约数、计数器回退时不报放量。Polymarket 保留美元成交量追踪，不用挂单流动性冒充持仓量计算换手率。
  - SQLite 自动新增原始合约数列，保留旧数据；旧行不反推合约数，至少积累 3 条有效历史快照后，下一次运行才能检测 Kalshi 放量。
  > 首次运行无历史时返回空，属正常，随运行逐步积累。

**跨平台背离**（`cross_platform.match_markets`）
- 首选 **embedding 语义匹配**（DashScope `text-embedding-v3` 批量向量化 + 余弦相似度，
  阈值 `signals.embedding_cutoff`）；API 不可用时自动降级回文本相似度。
- 命题门禁：同 `primary_type`、同问题形态、有限同义词归一化后的标题一致，且已有到期日期一致。保留年份、阈值、单位、方向、否定词和时间关系；不再把“降息”和“利率高于某阈值”配对。
- 通过命题门禁后才比较相似度和概率差；embedding 高分不能绕过门禁。复杂同义改写可能漏配，候选对均标记 `needs_review`，完整结算规则仍需人工核对。`shape_mismatch_penalty` 仅兼容旧配置，不再放行不同形态。

利率建议区分加息（YES 高 → 偏鹰）与降息（YES 高 → 偏鸽）；维持利率不变、无加息、混合动作或 Fed 人事命题不自动生成升降息建议。no cuts 不再自动映射为加息建议。

**事件级去重与阈值系列**（`signals.dedupe_anomalies_by_event` / `threshold_series.summarize`）
- 同一 `event_key` 的互斥合约（如 Kalshi 「联邦基金利率上限 above 3.25%/3.75%/…」）
  在进 Agent 前只保留代表（`agent.per_event_max`），避免同事件霸占槽位诱发幻觉；
- 阈值阶梯的 YES 概率构成离散 CDF，线性插值出**隐含中位数**与两端概率，
  一行摘要进报告 + 附加到对应异动的 prompt 证据。

**噪音门禁**（采集后、信号前）
- 美国州级 + 各国次国家级选举（市长/地方议会…，央行人事盘不误伤）；
- 体育/娱乐/奖项盘；关键词不命中的无关盘；
- 建议层：24h 零成交的噱头/死盘、概率 < `suggest_tail_prob_floor` 的尾部彩票合约不出建议。

> **重要过滤**：`filter_valuable` 剔除已过期、零流动性、长尾死盘，避免「已结算历史盘被当成
> 共识极强信号」的噪音污染。

---

## 📤 样例输出

`examples/sample_output.json` 是一次真实运行（关键词 Iran/Fed/tariff，60 个有价值市场）的完整结果，结构如下（节选）：

```jsonc
{
  "timestamp": "2026-10-21 15:30:11",
  "platforms": ["polymarket", "kalshi"],
  "total_markets": 60,
  "top_markets": [
    {
      "market_id": "2589811",
      "platform": "polymarket",
      "question": "Will the Fed decrease interest rates by 25 bps after the October 2026 meeting?",
      "yes_probability": 0.004,
      "volume_total": 8900000.0,
      "volume_24h": 371800.0,
      "event_types": ["interest_rate"],
      "primary_type": "interest_rate",
      "question_shape": "event",
      "days_to_resolution": 12
    }
  ],
  "anomalies": [
    {
      "anomaly_type": "money_inflow+whale",
      "severity": "high",
      "current_probability": 0.125,
      "details": { "signal_reasons": ["24h 成交占累计 32.1%，资金正在涌入", "..."] }
    }
  ],
  "cross_platform_pairs": [],
  "judgments": [
    {
      "question": "Will no Fed rate cuts happen in 2026?",
      "anomaly_type": "whale",
      "current_probability": 0.965,
      "event_summary": "市场几乎定价 2026 年美联储不降息……",
      "event_category": "Monetary Policy",
      "category_cn": "货币政策",
      "impact_on_a_share": "中性",
      "impact_direction": { "A股": "中性偏多", "美元": "偏强", "大宗商品": "承压", "黄金": "承压" },
      "confidence": 0.82,
      "trade_bias": "短期偏谨慎：关注美元指数与美债收益率联动",
      "rationale": "当前概率 96.5%……",
      "risk_note": "通胀回落快于预期则降息路径可能重估"
    }
  ],
  "asset_suggestions": [
    {
      "event_type": "ceasefire", "direction": "high",
      "action": "平仓原油多头，买入被制裁国资产",
      "trigger": "US x Iran ceasefire continues through October 22? (Yes 96.0%)",
      "needs_review": false
    }
  ]
}
```

同时生成 `output/report_<时间戳>.md` 人读报告（成交额 Top、跨平台对照、AI 判断、异动明细、资产建议、风险提示）。

---

## ⚙️ 配置

所有阈值集中在 `config.yaml`：`runtime`（平台/关键词/成交量·流动性·到期过滤/上限）、
`data_sources`（端点、Kalshi 成交时间窗口天数）、`signals`（各类异动阈值）、`agent`（模型/端点/最多判断事件数）、
`storage`（SQLite 路径）、`output`。命令行 `--min-volume/--limit/--platforms/--keywords` 可临时覆盖。

---

## 🧪 测试

```bash
python main.py --selftest                 # 离线自检（分类/过滤/信号/跨平台/格式化）
pip install pytest && pytest -q           # 单元测试
```

---

## ⚠️ 风险提示

- 预测市场概率反映参与者集体预期，**不等于**事件必然发生。
- 资产建议由关键词规则 + Agent 定性判断生成，**落地前需人工核对**语义与标的映射（阈值型问题尤其需复核）。
- 持仓集中度基于 `/holders` 头部样本，非全体分布，做市商库存会干扰。
- 本项目为研究框架，**不构成投资建议**。

## 📄 License

[MIT](LICENSE)


## 报告可信度与离线复核

- 方向建议采用保守门禁：联合/条件命题、阈值盘、地域/人事主题、否定命题及低概率事件不直接生成相反方向建议；明确的高概率非零加息/降息事件保留。所有保留建议需人工复核，不代表收益预测。
- 核协议单独分类，不按核威胁映射。被拦截的候选保存在 `suggestion_exclusions`，不是无声丢弃市场数据。
- 中位数仅在有效单调阶梯覆盖 50% 时估计；否则 `median=null`，使用 `median_status` / `median_bound` 给出边界。非单调或重复阈值不估计。HIGH/LOW 表示期间最高/最低值，非期末现价。
- 动态信号优先进报告和 Agent，报告保留完整标题。静态成交规模、占比与头部持仓不证明净资金流入。
- AI 必填资产方向不能为空，置信度必须为有限的 0~1 数值；无来源数字、无历史依据的准确率和无动量依据的变化描述会被拒绝，原因保存在 `judgment_rejections`。通过自动筛查不等于已完成事实核验，模型置信度未经校准。
- 可以离线复核旧报告：`python review_report.py output/report_20261027_091918.json`。生成同目录 `_reviewed.json/.md`，保留原报告，不联网、不调用模型、不写快照。复核仅覆盖旧报告已保存的候选，不能替代完整重跑。
