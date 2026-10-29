"""识别并剔除「美国国内州级选举」市场。

定位：本项目关键词以 A 股外部驱动（地缘 / 宏观 / 金融）为主。美国各州的参议员 /
众议员 / 州长等**州级选举**属于美国内政，对 A 股没有已知传导路径；且当检索关键词为空
时，这类盘凭借巨大成交量霸占 Top 榜单与异动列表，淹没真正相关的地缘/宏观事件。

因此从采集后、有价值过滤前的位置把它们摘掉，但**保留总统大选 / 国会控制权**等
全国性事件（可能影响美元、财政、关税预期）。

判定思路（宁可漏判不误伤）：
- 州名 + 选举语境词 同时出现 → 州级选举；
- 或命中天然州级短语（Senate race / Governor / Gubernatorial / state legislature …）；
- 单独出现州名（如「Russia-Georgia」中的 Georgia）不带选举词，不会被误剔除。
"""

from __future__ import annotations

import re

# 50 州 + 华盛顿特区
US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey",
    "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont", "virginia",
    "washington", "west virginia", "wisconsin", "wyoming",
    "washington dc", "washington, dc", "district of columbia",
}

# 天然的州级选举短语（出现即判定，无需州名）
_STATE_LEVEL_PHRASES = re.compile(
    r"\b("
    r"senate race|senatorial race|house race|congressional race|"
    r"governor'?s race|gubernatorial|state senate|state house|state assembly|"
    r"state legislature|state district|lieutenant governor|state attorney general|"
    r"attorney general race|secretary of state race|mayoral race|mayor of"
    r")\b",
    re.IGNORECASE,
)

# 选举语境信号（需与州名同时出现才判定）
_ELECTION_CONTEXT = re.compile(
    r"\b("
    r"election|voter|ballot|nominee|nomination|primary|caucus|"
    r"polling average|real clear|outperform|poll|polls|"
    r"win the seat|win the race|democrat|republican|dfl|gop|candidate|"
    r"governor|senator|mayor|prime minister|house of representatives"
    r")\b",
    re.IGNORECASE,
)


def _has_state(low: str) -> bool:
    return any(re.search(r"\b" + re.escape(s) + r"\b", low) for s in US_STATES)


def is_us_state_election(question: str) -> bool:
    """True 表示这是美国州级选举盘，应从管道中剔除。"""
    if not question:
        return False
    low = question.lower()
    if _STATE_LEVEL_PHRASES.search(low):
        return True
    if _has_state(low) and _ELECTION_CONTEXT.search(low):
        return True
    # 「race in <州>」结构（Senate/House/Governor race in Texas）
    if re.search(r"\brace in\b", low) and _has_state(low):
        return True
    return False


# 次国家级公职词（不限国别）：市长/市议会/县长/省长/州议会/地方区划等
_SUBNATIONAL_OFFICE = re.compile(
    r"\b("
    r"mayor|mayoral|city\s+(?:council|hall|magistrate)|county|councilman|borough|"
    r"governor|governorship|gubernatorial|lieutenant\s+governor|"
    r"state\s+(?:senate|house|assembly|legislature|district)|"
    r"provincial|prefectural|prefecture|regional\s+assembly|devolution"
    r")\b",
    re.IGNORECASE,
)

# 选举语境词（范围比 _ELECTION_CONTEXT 宽，但必须与公职词同现）
_ELECTION_WORDS = re.compile(
    r"\b(election|elect|race|ballot|vote|voter|nominee|nomination|primary|caucus|win)\b",
    re.IGNORECASE,
)

_FED_RE = re.compile(r"\bfed\b|federal reserve|fomc", re.IGNORECASE)
_GOV_WORDS = re.compile(r"govern\w*|gubernatorial", re.IGNORECASE)


def is_local_election(question: str) -> bool:
    """True 表示次国家级选举盘（任何国家的市长/地方议会/省长等）。

    与州级选举同理，对 A 股宏观传导信息为零（台北市长选举、韩国地方选举…）。
    注意保护「Fed governor」这类央行人事题：命中 fed 语境时先洗掉 govern 类词。
    """
    if not question:
        return False
    low = question.lower()
    if _FED_RE.search(low):
        low = _GOV_WORDS.sub("centralbanker", low)
    return bool(_SUBNATIONAL_OFFICE.search(low)) and bool(_ELECTION_WORDS.search(low))


# 体育 / 娱乐 / 文学奖项类市场：与政治·经济·财经无关，任何运行都直接剔除。
# （Kalshi 开放市场宇宙里棒球赛程/退役/格莱美/乡村音乐奖等常年占量，必须拦在管道前）
_OFF_THEME = re.compile(
    r"\b("
    r"nba|nfl|mlb|nhl|uefa|fifa|world cup|super bowl|stanley cup|world series|rookie|"
    r"games?\s+(?:be\s+played|played)|retirement|retire[sd]?|"
    r"album|single\b|song\b|film|movie|box office|grammy|oscar|emmy|cma|"
    r"country music|entertainer|vocalist|artist\b|eurovision|billboard|of the year|"
    r"awards?\b|nominations?\b|poems?\b|essays?\b|booker|pulitzer|national book"
    r")\b",
    re.IGNORECASE,
)


def is_off_theme(question: str) -> bool:
    """True 表示属体育/娱乐/奖项类噪音，与政经财经无关。"""
    return bool(question) and bool(_OFF_THEME.search(question))
