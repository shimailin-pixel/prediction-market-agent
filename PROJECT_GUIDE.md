# Prediction Market Agent —— 项目详细介绍

> 一句话定位：从 **Polymarket** 与 **Kalshi** 两个预测市场的公开 API 实时采集数据，
> 经过一套「过滤 → 分类 → 信号 → 异动 → 跨平台背离」的分析框架，再用一个 **LLM Agent 判断层**
> 把海外事件的概率变化翻译成对 **A 股 / 美元 / 大宗商品 / 黄金** 的方向性研判，最终产出 JSON + Markdown 报告。

本文面向想彻底读懂（或接手维护）这个项目的读者，按「为什么这样设计 → 架构 → 每个模块的细节 → 算法与阈值 → 局限」的顺序展开。

---

## 1. 核心理念

### 1.1 预测市场是「用真金白银投票」的概率引擎
预测市场的每个合约都是一个「某事件会不会发生」的二元命题，价格即市场公认的发生概率（0~1）。
与传统民调、卖方研报相比，它有三个独特价值：
- **实时**：价格随新闻秒级重定价；
- **激励相容**：参与者押的是真钱，噪声比免费问卷小；
- **前瞻**：很多地缘 / 宏观事件在其它媒体还没反应时，盘口已先动。

### 1.2 目标不是「预测事件」，而是「翻译成资产方向」
项目服务的场景是 **A 股资产配置的海外外部驱动监测**：美联储利率、地缘冲突、关税、原油、美元流动性。
因此所有中间产物都朝一个终点收敛 ——「这件事对 A 股 / 美元 / 大宗 / 黄金 是利好还是利空」。
这也解释了为什么会有大量「噪音门禁」：与这个目标无关的市场（体育、娱乐、州级选举）会被坚决剔除。

### 1.3 一条贯穿始终的工程哲学：宁可漏，不可假
无论是跨平台配对、信号阈值，还是 Agent 事实字段由代码填充、模型只填解读，
设计原则都是 **控制假信号**。项目文档里明确写了「诚实的局限」，见第 11 节。

---

## 2. 技术栈与依赖

| 维度 | 选型 | 说明 |
| --- | --- | --- |
| 语言 | Python 3.10+ | 全项目 `from __future__ import annotations`，大量类型标注 |
| HTTP | `httpx` 优先，缺失自动回落标准库 `urllib` | 见 `src/http.py`，零强依赖 |
| 数据模型 | `dataclass`（内部流转）+ `pydantic`（Agent 输出校验） | Market/Anomaly 用 dataclass，Judgment 用 pydantic |
| 配置 | `PyYAML` + `python-dotenv` | `config.yaml` 主配置，`.env` 放密钥 / 代理 |
| LLM | OpenAI 兼容端点（默认阿里云 DashScope 通义千问 `qwen-plus`） | `openai>=1.0.0` SDK |
| 向量 | DashScope `text-embedding-v3` | 跨平台语义匹配，可选、可降级 |
| 存储 | SQLite（标准库 `sqlite3`） | 概率 / 成交量历史快照，零依赖 |
| 测试 | `pytest` + 内置离线 `--selftest` | 双模式：不联网自检 + 联网实跑 |

`requirements.txt` 只有 6 个核心依赖，且网络 / LLM 相关的全部是「有则更好、无则降级」，
保证在**没有密钥、没有 httpx、没有代理**的最差环境里，仍能跑通「数据 + 信号」骨架。

---

## 3. 整体架构

### 3.1 四层结构

```
┌─────────────────────────────────────────────────────────────┐
│  编排层  main.py    run_pipeline：把下面四层串成一条流水线      │
└─────────────────────────────────────────────────────────────┘
        │
   ┌────▼─────────────────────────┐   采集：Polymarket / Kalshi 公开 API
   │  数据接入层  data_source/      │   归一化：差异极大的原始字段 → 统一 Market
   │   polymarket / kalshi         │   增强：概率历史 / 持仓集中度
   │   storage(SQLite) / embeddings│   存储：历史快照供跨运行对比
   └────┬─────────────────────────┘
        │  List[Market]
   ┌────▼─────────────────────────┐   classify / elections / signals /
   │  分析框架层  analysis/         │   tracker / cross_platform / threshold_series
   │   过滤 → 分类 → 信号 → 异动 →  │   产出：List[Anomaly] + 阈值摘要 + 资产建议
   │   跨平台背离 → 阈值隐含分布     │
   └────┬─────────────────────────┘
        │  List[Anomaly]（事件级去重后）
   ┌────▼─────────────────────────┐   prompt：多视角推理脚手架
   │  Agent 判断层  agent/          │   LLM：填「解读与方向」，不碰数字
   │   judge：pydantic 结构化校验    │   产出：List[Judgment]
   └────┬─────────────────────────┘
        │
   ┌────▼─────────────────────────┐   Markdown 报告 + 终端摘要
   │  输出层  report.py             │   + JSON（机器可读）
   └──────────────────────────────┘
```

### 3.2 流水线精确步骤（对应 `main.py: run_pipeline`）

1. **采集** `fetch_all`：按 `runtime.platforms` 调用各平台 Client；
2. **分类** `classify.annotate`：每个 Market 就地填充 `event_types / primary_type / question_shape`；
3. **选举噪音剔除** `elections.is_us_state_election` + `is_local_election`（可开关）；
4. **相关性门禁** `signals.filter_relevant`：体育/娱乐/奖项无条件剔除，其余须命中关键词或分类体系；
5. **有价值过滤 + 配额截断** `filter_valuable`（剔过期/零流动性/长尾）→ `quota_truncate`（每平台保底）；
6. **增强** `enrich_polymarket`：对 PM 成交额 Top N 补概率历史 + 持仓集中度；
7. **信号与异动**：`detect_smart_money` + `tracker.track_changes` + `cross_platform.match_markets`；
8. **阈值系列聚合** `threshold_series.summarize`：阶梯合约折成隐含分布，并把摘要作为证据挂回异动 `details`；
9. **Agent 判断**：先 `dedupe_anomalies_by_event`（同事件互斥合约只留代表），再 `judge.judge_batch`；
10. **资产建议** `generate_asset_suggestions`；
11. **落快照** `SnapshotStore.save_batch`（供下次运行对比）→ 组装 payload → 写报告。

> 关键顺序：**过滤都在增强/信号之前**，避免对死盘、过期盘、无关盘浪费网络请求和算力，也避免它们污染统计。

---

## 4. 目录结构

```
PredictionMarket-Agent/
├── main.py                     # CLI 入口 + run_pipeline 编排 + selftest 离线自检
├── config.yaml                 # 全部阈值 / 端点 / 关键词（单一事实来源）
├── requirements.txt
├── .env / .env.example         # DASHSCOPE_API_KEY / HTTPS_PROXY
├── data/                       # SQLite：snapshots.db（历史快照）
├── output/                     # report_<时间戳>.json + .md
├── tests/test_analysis.py      # pytest 离线单元测试
└── src/
    ├── config.py               # YAML + .env 加载、点分读取、路径锚定、命令行覆盖
    ├── models.py               # Market / Anomaly 统一数据模型
    ├── http.py                 # HttpClient：httpx 优先→urllib 降级，重试/退避/代理
    ├── util.py                 # safe_float / 日期 / 金额格式化 / 短标题 / 会议尾巴抽取
    ├── report.py               # Markdown 报告渲染 + 终端摘要
    ├── data_source/
    │   ├── polymarket.py       # Gamma 检索 + CLOB 概率历史 + Data 持仓集中度
    │   ├── kalshi.py           # /markets + 成交时间窗口分段扫描（2026 字段迁移适配）
    │   ├── storage.py          # SQLite 快照存储（追加语义）
    │   ├── embeddings.py       # DashScope 批量文本向量化 + 余弦相似度
    │   └── __init__.py         # 导出各 Client
    ├── analysis/
    │   ├── classify.py         # 事件类型正则 + 优先级 + 问题形态 + 资产映射表
    │   ├── elections.py        # 州级/次国家级选举 + 体育娱乐奖项 判定
    │   ├── signals.py          # 过滤 / 聪明钱信号 / 事件去重 / 资产建议
    │   ├── tracker.py          # prob_spike / volume_spike（对比历史快照）
    │   ├── cross_platform.py   # 跨平台匹配（embedding/文本）+ 背离五重门禁
    │   └── threshold_series.py # 阈值阶梯 → 隐含分布（中位数线性插值）
    └── agent/
        ├── prompt.py           # SYSTEM + JUDGE prompt + 证据拼装（数字全代码填）
        └── judge.py            # EventJudge：LLM 调用 + pydantic 校验 + 平台多样化选取
```

---

## 5. 数据接入层（`src/data_source/`）

### 5.1 Polymarket（`polymarket.py`）—— 三个公开 API，均无需认证

| API | 端点 | 作用 |
| --- | --- | --- |
| **Gamma** | `/public-search`（回落 `/events?search=`） | 关键词检索 → events → markets，拿问题/概率/成交额/流动性 |
| **CLOB** | `/prices-history` | YES token 概率时间序列 → 24h/7d 概率动量 |
| **Data** | `/v2/holders`（回落 `/holders`） | 头部持仓集中度 → 识别巨鲸 |

解析要点（都是踩过的坑）：
- **按 outcomes 标签定位 Yes/No**，不硬取 `prices[0]/[1]`；非二元（多结果）市场直接拒绝解析，避免把「伊朗哪位候选人上台」这种多选项算成 Yes/No；
- 缺失 `outcomePrices` 时用 `bestBid/bestAsk` 兜底；
- 请求节流 `throttle=0.25s`，避免打爆公开接口；
- 增强只对「成交额 Top N」做（`runtime.enrich_top_n`），因为 holders / prices-history 是逐市场请求，全量做太慢。

### 5.2 Kalshi（`kalshi.py`）—— 最「叛逆」的一个数据源

Kalshi 的采集逻辑是整个项目里工程决策最密集的地方，原因有四条实测结论：

1. **端点迁移**：2026 Q1 生产域名迁到 `external-api.kalshi.com/trade-api/v2`，公开 `/markets` 无需认证。
2. **`query` 参数被 API 静默忽略**：实测 `query=Iran` 和 `query=Fed` 返回完全相同 → **服务端关键词检索不可用**。因此关键词精筛全部放到本地门禁（`filter_relevant`）做，采集层只负责「按窗口拉全量」。
3. **默认排序是「成交时间从远到近」**：长窗口下前几十页全是 378+ 天的远期选举/体育盘，近端的 Fed/宏观盘永远翻不到。解决方案是 **"近细远粗"分段扫描**（`_segments`）：
   - `near_horizon_days`（默认 60 天）内每 `near_segment_days`（7 天）切一段；
   - 之外每 `far_segment_days`（30 天）切一段；
   - 每段用 `min_close_ts/max_close_ts` 窗口 + cursor 翻页，`max_pages_per_segment`（4）控请求量。
   - 这一招还**天然排掉了 KXMVECROSSCATEGORY 串关组合**（`_is_parlay` 过滤），因为串关到期结构不同。
4. **字段迁移**：整数字段（`yes_bid`/`volume`/`liquidity`）已移除，改为字符串定点（`yes_bid_dollars`/`volume_fp`/`open_interest_fp`），全部走 `safe_float` 解析。

**美元口径**：1 张合约到期支付 $1，故 `成交额(USD) ≈ 成交合约数 × YES 概率`（`_usd_volume`）。
这导致 Kalshi 的美元量级天然小于 Polymarket 真实 USD —— 这个量级差异是后续多处「分平台基准」设计的根源（见 6.3、3 步骤 5）。

- YES 概率：优先取 `(yes_bid + yes_ask)/2` 中间价，回落 `last_price_dollars`；
- 流动性：用 `open_interest_fp × prob`（未平仓合约美元）作为流动性代理。

### 5.3 快照存储（`storage.py`）

- 表 `snapshots(market_id, platform, ts, yes_probability, volume_total, volume_24h, liquidity)`，主键 `(market_id, ts)`；
- **追加语义**：同一 market_id 每次运行写一行（`INSERT OR REPLACE`），按 ts 升序，形成时间序列；
- 提供 `history()`（取全部历史）和 `previous_snapshot()`（取某时刻前最近一条），供 `tracker` 做相邻对比；
- 首次运行无历史 → 动态异动返回空，属正常，随运行逐步积累。

### 5.4 向量化（`embeddings.py`）

- 复用 Agent 同一份 `DASHSCOPE_API_KEY` 和 base_url，走 OpenAI 兼容 `/embeddings`；
- DashScope `text-embedding-v3` 单批上限 **10 条**，代码分批请求；
- **任何失败都返回 None**，由调用方 `cross_platform` 降级回文本相似度 —— 网络失败 / 限额 / 无 key 都不打断主流程。

---

## 6. 分析框架层（`src/analysis/`）

### 6.1 事件分类（`classify.py`）

把一句自然语言问题映射成结构化标签，是后续一切解读的地基。

- **词边界正则**（`TYPE_PATTERNS`）：`toward` 里的 `war` 不会被误判 —— 所有关键词都用 `\b...\b`；
- **多命中取主类型**（`TYPE_PRIORITY`）：优先级设计有讲究 ——
  `机制类(ceasefire/nuclear/war/tariff/sanctions) > 宏观类(interest_rate/recession/inflation) > 资产类(oil/crypto/equity_index) > 地理类(iran/mideast/russia/china)`。
  逻辑：**交易动作由事件性质主导，而非发生地点**（「停火」比「伊朗」更能决定怎么做）。
- **问题形态**（`classify_question_shape`）：
  - `threshold`：含 above/below/hit/reach… → 是「价格/水平」类，如「油价是否高于 $110」；
  - `event`：will/does… 引导 → 是「会不会发生」；
  - 这个区分至关重要：**阈值型和事件型的概率含义不可直接比较**，跨平台匹配、建议分桶都据此设卡。
- **资产映射表**（`EVENT_ASSET_MAP`）：每个事件类型 × {high/low/mid}（事件型）或 level_high/level_low（阈值型）→ `{action, reason}`。这是规则层「资产配置建议」的字典。

### 6.2 选举与主题噪音（`elections.py`）

三道本地过滤器，都遵循「宁可漏判不误伤」：
- `is_us_state_election`：州名 + 选举语境词同现，或命中天然州级短语（Senate race/Governor…）才判；「Russia-**Georgia**」里的Georgia 无选举词 → 不误伤；
- `is_local_election`：次国家级（市长/地方议会/省长…），任何国家都剔；特别保护 **「Fed governor」这类央行人事题** —— 命中 fed 语境时先把 `govern*` 词替换掉再判；
- `is_off_theme`：体育/娱乐/文学奖项（NBA/Grammy/Pulitzer…）无条件剔，因为 Kalshi 开放市场宇宙里这类常年占量。

### 6.3 信号层（`signals.py`）—— 项目的分析心脏

**（1）`filter_valuable` 有价值过滤**
剔除：已过期（`days_to_resolution<0`）、零流动性（`<min_liquidity`）、长尾（`volume_total<min_volume`）。
**支持按平台覆盖阈值**：`runtime.per_source.kalshi.min_volume=2000`（因 Kalshi 量级偏小，见 5.2）。
这一步从源头收住「63% 是已结算历史盘、信号被 0%/100% 噪音淹没」的老问题。

**（2）`quota_truncate` 平台保底截断**
按成交额全局排序截断时，为每个平台保底 `min_per_platform`（默认 20）个名额。
防止 Kalshi 在 Polymarket 真实 USD 量级面前被完全淹没、连榜单都进不去。

**（3）`detect_smart_money` 聪明钱静态信号**（当前快照即可判定）

| 标签 | 触发条件（默认阈值） | 含义 |
| --- | --- | --- |
| `money_inflow` | 24h 成交 / 累计 ≥ `volume_24h_share`(0.08) | 资金正在涌入 |
| `volume` | 累计成交 > **本平台**均值 × `volume_multiple`(3.0) | 放量（分平台基准，见下） |
| `prob_momentum` / `prob_trend` | 24h/7d 概率变动 ≥ `prob_change_24h_pts`(5)/`_7d`(10) | 概率重定价（来自 CLOB 历史） |
| `whale` | 最大单一钱包占可见头部 ≥ `concentration_top1_pct`(45) | 巨鲸主导（来自 /holders） |

命中 ≥2 类记 `high`，否则 `medium`。
**放量基准按平台各自计算**：`avg_by_plat` —— 否则 Kalshi 永远达不到被 Polymarket 真实美元拉高的全场均值。

**（4）`track_changes` 动态异动**（`tracker.py`，需历史快照）
- `prob_spike`：当前概率相对上一快照跳变 ≥ `prob_spike_pts`(15) 点；
- `volume_spike`：本次累计成交增量 ≥ 历史周期增量均值 × `volume_multiple`(3.0)。

**（5）`dedupe_anomalies_by_event` 事件级去重**
同一 `(platform, event_key)` 的互斥合约（Fed 利率阶梯、伊朗候选人系列）在**进 Agent 前**只留代表
（`severity` 优先、同级取成交额大者）。原因：同事件多个合约是一个概率分布的切片，全送进 LLM 会把槽位占满、诱发「同一件事解读 8 遍」的幻觉。**报告仍展示全部异动，只收窄 Agent 输入。**

**（6）`generate_asset_suggestions` 资产配置建议**
按 `primary_type` 聚合、查 `EVENT_ASSET_MAP`，三道防噪闸：事件聚合 + 尾概率下限 + 24h 活跃门槛。

> **重点：`_rate_bucket` 鹰派强度分桶**（这是本项目对「建议方向反了」这类错误的修复）
> 利率盘不能简单按 YES% 高低分桶，因为命题方向可能是反的：
> - 肯定式「rate **cut** 70%」→ 鹰派强度 = 100−70 = 30 → **low 桶**（降息预期，利好成长股）；
> - 否定式「**no** rate **cuts** 96%」→ 鹰派强度 = 96 → **high 桶**（不降息=偏鹰，利好银行股）；
> - 阈值型「rate **above** 3.75%」→ 概率本身即利率高于某值的定价 → 用原始概率。
> 识别靠三个正则：`_NEGATED_NO_CUT`（no/not/neither … cut/hike/change）、`_NEGATED_HOLD`（hold/pause）、`_CUT_HIKE`。
> **尾概率门禁按「事件侧」概率判**：0.1% 的降息彩票盘即使折出鹰派 99.9 也不出建议；但「No cuts 96%」是真实鹰派共识，保留。

> **needs_review 规则**：`question_shape=threshold` 一律需复核；
> 且 `war/iran/russia/china/mideast` 的 **low 桶**一律需复核 —— 因为这些是「入侵/冲突没发生」的低概率，
> 「低概率 = 风险消退」只是**现状推论**，不是缓和事件本身（1.8% 的「中国入侵台湾」不等于可以买入风险资产）。

### 6.4 跨平台匹配与背离（`cross_platform.py`）

目标：同一件事在两个平台措辞不同（「Fed 降息 25bp」 vs 「联邦基金利率上限高于 4%」），要把它们配对，
配对成功后算概率背离，背离大 = 一个平台可能错误定价 = 可交易线索。

**匹配方式（双轨）**：
- 首选 **embedding 语义相似度**（DashScope 向量 + 余弦），能配上「措辞结构不同但语义同」的盘；
- 无密钥 / 调用失败 → 自动降级回 **归一化文本 SequenceMatcher 相似度**。

**五重门禁（全部在 `match_markets` 内强制执行，展示层不再放宽）**：
1. `primary_type` 必须相同；
2. **央行主体冲突拒配**：`_SUBJECT_CONFLICTS` 九组（fed/boj/ecb/boe/snb/pbo/rbi/boc/anz），两侧标题合并命中 >1 组即拒 —— 修的是「Fed 加息 vs 日本央行加息」字面高相似却南辕北辙的误配；
3. **形态可比性**：任一侧 `question_shape=unknown` 直接拒配；事件型 vs 阈值型则抬高 cutoff（`shape_mismatch_penalty`）；
4. **相似度门槛**：文本 ≥ `similarity_cutoff`(0.62) / embedding ≥ `embedding_cutoff`(0.80)，各自独立阈值；
5. **背离 ≥ `cross_platform_divergence_pts`(10) 点**：不足门禁的配对**不返回、不进报告**。

> 差点对上线的分数会记进 `near_miss` 日志，供实跑校准 cutoff。

### 6.5 阈值系列隐含分布（`threshold_series.py`）

Kalshi 的「联邦基金利率上限 above 3.25%/3.50%/3.75%/… 是否成立」是同一事件的**一排互斥阈值合约**，
它们的 YES 概率构成一条**离散累积分布函数（CDF）**。逐行展示 8 个合约信息密度极低，
这里折成一行摘要：
- `_extract`：从标题抽 (方向 above/below, 阈值, 单位)；
- `_median`：在离散 CDF 上**线性插值**求中位数（0.5 分位点）；
- `format_line`：输出「中位数 + 两端概率」，美元符号前置（`$105` 而非 `105$`）；
- `_label`：系列名 = 数字前缀 + 方向（≥/≤）+ 会议尾巴（`meeting_hint` 抽取「following … meeting」），
  解决同前缀不同会议/方向的多个系列在报告里显示成同一行的问题。
摘要既进报告，也作为结构化证据附到同事件代表市场的 Agent prompt 里（LLM 拿到完整分布而非单点）。

---

## 7. Agent 判断层（`src/agent/`）

### 7.1 设计铁律
参考 [TradingAgents](https://github.com/TauricResearch/TradingAgents) 的「多视角辩论」思路，压缩成单次判断：
> **所有数字（概率/变化/成交额）由代码写进 prompt，模型不得臆测数值；模型只产出「解读与方向」。**

这条把 LLM 的「算术/编造」风险降到最低，只保留它擅长的「语义关联与方向判断」。

### 7.2 四视角推理（`prompt.py`）
JUDGE_PROMPT 引导模型依次从：
① **基础概率**（市场当前定价说明什么共识）
② **动量**（概率/资金变化）
③ **跨平台定价**（是否有另一平台对照验证）
④ **风险传导**（对 A 股外部驱动的具体路径）
再下结论。并注入**当前日期**（防止把已过去年份当「未来」风险的时间幻觉）。

### 7.3 结构化输出（`judge.py`）
`Judgment`（pydantic）字段：

| 来源 | 字段 |
| --- | --- |
| 代码填（事实） | `market_id / platform / question / anomaly_type / current_probability / timestamp` |
| LLM 填（解读） | `event_summary / event_category / category_cn / impact_on_a_share / impact_direction{A股,美元,大宗商品,黄金} / confidence(0~1) / trade_bias / rationale / risk_note` |

- 走 OpenAI 兼容端点，`response_format=json_object`（gpt/qwen 系），强制严格 JSON；
- **`impact_direction` 四资产键必填**，展示层再用 `_v()` 兜底防空串；
- 无 key / 调用失败 → 返回 None，主流程（数据 + 信号）不受影响；
- `select_diverse`：每平台保底 `min_per_platform`(2) 个名额进 Agent，防止 high 级全被 Polymarket 占据、Kalshi 的 medium 永远进不了判断。

---

## 8. 输出层（`src/report.py`）

Markdown 报告固定分节，每节都对应流水线的一类产物：

1. **头部统计**：生成时间 / 数据源 / 关键词 / 有价值市场数 / 异动数 / AI 判断数 / 建议数；
2. **成交额 Top 市场**（**分平台各列前 10**，避免 Kalshi 在全局排序里隐身）；
3. **阈值系列隐含分布**；
4. **跨平台同事件对照**（背离门禁后的配对）；
5. **Agent 判断层**（结构化解读，逐条展示）；
6. **异动信号明细**（Top 30 + 「其余 N 条见 JSON」说明）；
7. **资产配置建议**（含「需复核」列）；
8. **风险提示**（固定免责）。

标题统一走 `short_question(text, limit)`：**超长时保留尾部**（「头…尾」），
因为 Kalshi 阶梯盘的区分信息（阈值 + 会议日期）常在长标题中部/尾部，纯头部定长截断会让多行撞脸。
另有 `print_summary` 输出精简终端摘要。

---

## 9. 配置系统（`src/config.py` + `config.yaml`）

### 9.1 关键工程细节：路径锚定
`PROJECT_ROOT = src/ 的上一级`。`resolve_config_path` / `project_path` 保证**从任意目录启动**时，
配置和数据 / 输出路径都落在项目内同一处，不因 cwd 变化而漂移（Windows 上双击运行、IDE 运行都受益）。

### 9.2 密钥优先级
`get_llm_credentials`：`DASHSCOPE_API_KEY > OPENAI_API_KEY > AGENT_API_KEY`；base_url/model 同理有环境变量覆盖。
api_key 为空 = 应跳过 LLM 判断层。

### 9.3 配置分区速查（`config.yaml`）

| 区 | 关键项（默认值） |
| --- | --- |
| `runtime` | platforms / keywords(12 个地缘宏观词) / `min_volume`(10000) / `min_liquidity`(1) / `per_source.kalshi.min_volume`(2000) / `min_per_platform`(20) / `max_days_to_expiry`(400) / `enrich_top_n`(12) / `limit`(300) / `suggest_min_volume_24h`(500) / `suggest_tail_prob_floor`(2.0) |
| `data_sources.polymarket` | gamma/clob/data base |
| `data_sources.kalshi` | base(external-api) / `max_close_days`(400) / `near_horizon_days`(60) / `near_segment_days`(7) / `far_segment_days`(30) / `max_pages_per_segment`(4) |
| `signals` | `volume_multiple`(3.0) / `volume_24h_share`(0.08) / `prob_change_24h_pts`(5) / `prob_change_7d_pts`(10) / `prob_spike_pts`(15) / `concentration_top1_pct`(45) / `cross_platform_divergence_pts`(10) / `similarity_cutoff`(0.62) / `use_embedding`(true) / `embedding_cutoff`(0.80) / `shape_mismatch_penalty`(0.05) |
| `agent` | `model`(qwen-plus) / base_url / `embed_model`(text-embedding-v3) / `temperature`(0.1) / `max_events`(8) / `min_per_platform`(2) / `per_event_max`(1) |
| `storage` | `sqlite_path`(data/snapshots.db) |
| `output` | dir(output) / formats([json, markdown]) |

命令行 `--platforms/--keywords/--min-volume/--limit/--no-llm/--no-enrich` 经 `apply_overrides` 临时覆盖。

---

## 10. 运行与测试

```bash
# 安装
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # 填 DASHSCOPE_API_KEY / HTTPS_PROXY

# 双模式测试
python main.py --selftest          # 离线自检：不联网跑通分类/过滤/信号/跨平台/格式化（30 项断言）
pytest -q                          # 单元测试：覆盖配对门禁、建议分桶、格式化等回归用例

# 实跑
python main.py                                       # 默认双平台 + 内置关键词
python main.py --keywords "Iran,Fed,tariff"
python main.py --platforms polymarket                # 单平台
python main.py --no-llm                              # 只跑数据+信号
python main.py --no-enrich                           # 跳过 CLOB/Data 增强（更快）
```

> **双模式验证规范**：任何改动先过 `--selftest` + `pytest`（离线，秒级），再实跑一次看联网效果。
> 跨平台 cutoff、放量倍数等阈值参数「先调一次、实跑验证再定」，不凭拍脑袋定死。

产物：`output/report_<时间戳>.json`（含完整 `market` 上下文，机器可读）与 `.md`（人读）。

---

## 11. 诚实的局限与设计取舍

项目文档明确列出以下局限，使用与二次开发时都应知道：

- **语义 ≠ 命题等价**：embedding 只能判断「语义接近」，判断不了「两个合约是不是同一个可对比的量」。
  同事件不同行权价/阈值的盘可能靠得很近 —— 所以保留形态惩罚 + 主体冲突 + 背离门禁，宁缺毋滥。
- **事件型 vs 阈值型概率不可直接相减**：即使配对成功，二者的「概率」衡量的是不同的东西，
  背离信号对同形态才有强意义。
- **持仓集中度只基于 /holders 头部样本**：非全体分布，做市商会把头部持仓摊薄，`top1_pct` 可能低估。
- **Kalshi 美元口径是估计值**：`合约数 × 概率`，与 Polymarket 真实 USD 不同量级 → 一切「放量/榜单」都分平台算。
- **概率跳变需要历史**：`prob_spike/volume_spike` 首次运行返回空，要连续跑（或持续模式）才积累得出。
- **预测市场概率 ≠ 事件必然发生**；资产建议由「关键词规则 + Agent 定性」生成，**落地前必须人工核对**，本报告不构成投资建议。

---

## 12. 近期演进要点（可作为「为什么代码长这样」的背景）

1. **跨平台防误配升级**：新增央行主体冲突门禁（修 Fed 盘与日本央行盘被 embedding 高分误配）；
   形态不可比（unknown）直接拒配；并把「背离 ≥ 门禁」同时作用于配对返回 —— 弱背离噪音不再进报告。
2. **利率建议按鹰派强度分桶**：`_rate_bucket` 修掉了「No rate cuts 96% 被当成降息利多」的方向反转。
3. **地缘低概率建议标需复核**：`_STATUS_QUO_LOW_TYPES` 让「入侵/冲突没发生」类 low 桶建议必须人工核对。
4. **Agent 时间幻觉治理**：prompt 注入当前日期 + 四资产必填；展示层 `_v()` 兜底空值。
5. **报告可读性**：Kalshi 阶梯盘「头…尾」智能截断 + 阈值系列名附方向与会议尾巴，解决多行撞脸。

---

## 13. 一图总结数据流

```
关键词 ─▶ [Polymarket Gamma] + [Kalshi 分段扫描]
                        │  原始 JSON
                        ▼
                 归一化 Market（USD / 0~1 概率）
                        │
        classify（事件类型+形态）  elections（剔选举）  filter_relevant（相关性门禁）
                        │
        filter_valuable（剔过期/死盘/长尾）  quota_truncate（平台保底）
                        │
        enrich（PM：CLOB 概率历史 + Data 持仓）
                        │
   detect_smart_money（聪明钱）  tracker（prob/volume 跳变）  cross_platform（背离）
                        │
        threshold_series（阶梯→隐含分布，证据回填）
                        │
        dedupe_by_event ─▶ EventJudge（LLM 多视角，pydantic 校验）
                        │
        generate_asset_suggestions（规则 + 鹰派分桶 + needs_review）
                        │
        SnapshotStore.save_batch（写 SQLite，供下次对比）
                        ▼
        report.py ─▶ output/report_<ts>.json + .md  ─▶ 终端摘要
```

---

**License：MIT**（见 `LICENSE`）。本框架为研究用途，所有输出不构成投资建议。
