# CS2 数据源方案（HLTV 为主）

> 调研日期：2026-08-22，含生产环境实测。目标：为 Polymarket CS2 交易 bot 提供选手/队伍数据管道。

## 结论（推荐栈，按可靠性×成本排序）

1. **bo3.gg 公开 JSON API（首选，免费）** -- 赛程/赛果/排名/队伍档案/赔率，字段干净无鉴权（灰色：无官方授权，须抽象接口防关门）
2. **Liquipedia MediaWiki API（免费、最合规）** -- 交叉验证与低级别赛事补充，严格限速
3. **HLTV 自爬（curl_cffi 模拟 Chrome TLS 指纹）** -- 仅拉 bo3.gg 没有的特征（Rating 2.1/ADR/KAST/Impact/地图胜率），≥2s 限速 + 日缓存
4. PandaScore 免费档 -- 仅作第三赛程源；⚠️ 免费档条款禁止博彩用途，实盘前排除
5. 不推荐：GRID/Bayes（企业销售）、GOTV demo 解析（仅回合级特征才需要）

## 1. HLTV.org

- **无官方 API**，ToS 禁爬。实测直接抓 `/matches` 返回 403（Cloudflare）。
- 突破手段：`curl_cffi`（`impersonate="chrome"`）、Playwright 或付费解锁代理（ZenRows/ScraperAPI ~$50/月）。即使突破也须 ≥2s/请求 + 本地缓存，否则封 IP。
- URL 模式（稳定多年）：
  - 赛程 `/matches`；结果 `/results?startDate=&endDate=&requireAllTeams=2`（offset 分页）
  - 排名 `/ranking/teams/{yyyy}/{month}/{day}`（每周）
  - 玩家榜 `/stats/players?startDate=…&matchType=BigEvents`（Rating 2.1、K/D、ADR、KAST、Impact、APR、DPR；可按地图筛）
  - 单玩家 `/stats/players/{id}/{nick}`、逐场 `/stats/players/matches/{id}/{nick}`
  - 队伍地图胜率 `/stats/teams/maps/{mapId}/{teamId}/{team}`；H2H 在比赛页内嵌
- 现成库无一可靠：`gigobyte/HLTV`（TS，声明不再维护但仍有活动，选择器逻辑可参考）、`SocksPls/hltv-api`（Python，2022 停更）、其余半死。

## 2. bo3.gg（实测 2026-08 可直接 GET，无鉴权）

| 端点 | 内容 |
|---|---|
| `https://api.bo3.gg/api/v2/matches/finished?date=YYYY-MM-DD&filter[discipline_id][eq]=1` | 当日已结束比赛：比分、逐图、赔率、AI 预测、赛事层级、奖金池 |
| `https://api.bo3.gg/api/v2/matches/upcoming?date=YYYY-MM-DD&filter[discipline_id][eq]=1` | 当日赛程：stars、BO 类型、**40+ 博彩市场赔率（bet_updates）**、ai_predictions |
| `https://api.bo3.gg/api/v2/team_rankings?page=1&per_page=30&filter[discipline_id][eq]=1` | 队伍排名（meta.total_pages 翻页） |
| `https://api.bo3.gg/api/v1/teams/{slug}` | 队伍档案：世界排名、阵容（每人 six_month_avg_rating）、转会史、冠军、半年奖金 |
| `https://api.bo3.gg/api/v1/matches/{match_slug}/players_stats` | 单场玩家统计 |

- discipline_id=1 即 CS。0.3–0.5s 请求间隔礼貌限速。
- **bet_updates 博彩赔率可直接作 Polymarket 价格对比的"公平概率"基准**，解决基准模型冷启动。
- 风险：无 SLA、无授权，随时可能加 Cloudflare/鉴权 -> 代码必须抽象 `DataSource` 接口可切换。

## 3. Liquipedia

- MediaWiki API 免费开放：`https://liquipedia.net/counterstrike/api.php`（`action=parse`/`action=query`）。
- 限速：常规 ≤1 req/2s；`action=parse` ≤1 req/30s。必须自定义 UA（含联系方式），通用 UA 被封。禁爬 HTML 页，只能 API。CC-BY-SA 署名。
- LPDB/Cargo 结构化查询不可用，需申请（Discord，≤60 req/h）；比赛列表页是 Lua 渲染，需 `action=parse&prop=text` 拿渲染后 HTML 解析。
- Python 参考：`liquipediapy`（PyPI，2020 后未更新，需自修）。

## 4. 关键指标 → 数据源映射

| 指标 | 来源 |
|---|---|
| Rating 2.1 / ADR / KAST / Impact / DPR | 仅 HLTV |
| 队伍排名/积分 | HLTV、bo3.gg |
| 地图池 pick/ban/win% | HLTV、bo3.gg、Liquipedia |
| 近 3 个月状态 | 自建历史库自算（bo3.gg finished 回填） |
| H2H | 自算 或 HLTV 比赛页 |
| 赛事层级/奖金池 | bo3.gg tournament（tier s/a/b + stars） |
| 博彩赔率基准 | bo3.gg bet_updates |
| 回合级/经济特征（后期） | GOTV demo（demoparser2 / awpy） |

## 5. 架构建议

- 比赛发现主渠道：bo3.gg `matches/upcoming` 按天轮询（可按 stars/tier 只做 S/A 级）
- 历史库：bo3.gg `matches/finished` 按日回填 -> SQLite，自算 Glicko-2（分地图）、近 90 天状态、H2H
- HLTV 每日低频一次补充特征（curl_cffi，失败即降级跳过）
- 参考开源：[haidamykyta/cs2](https://github.com/haidamykyta/cs2) -- XGBoost + Glicko-2 + 校准 + Kelly，与本项目思路几乎相同，值得通读
