# GitHub 开源项目调研报告（Polymarket CS2 交易 Bot 参考）

> 调研日期：2026-08-22。四路并行调研：Polymarket 机器人 / CS2 预测 ML / demo 解析工具 / 体育博彩量化框架。
> 共考察约 100 个仓库，深读 20+ 个。全部核对活跃时间（旧 SDK 已死，还在用它的项目已剔除）。

---

## 〇、最重要的一条发现

**`0xAidan/esports-model`（2026-08 活跃）和我们目标是同一个：CS2 模型 + Polymarket +EV 扫描（signal only）**。它的数据源决策文档、防泄漏工程、Polymarket 端数学、六道流动性闸门可以直接参考。除此之外 GitHub 上没有任何 CS2 专用 Polymarket bot--生态位是空的。

同样重要的是反面教材：`haidamykyta/cs2` 回测 +304%，**无泄漏重测后只有 +2.26% ROI，实盘还亏了 27%**。防数据泄漏是这个项目的第一工程问题。

---

## 一、参考架构（四路共识收敛后的形态）

```
┌ 数据层 ─────────────────────────────────────────┐
│ 双 SQLite 库：match_stats.db / odds_market.db    │ ← kyleskom 模式
│ 数据源：bo3.gg(赛程/赛果/赔率) + Liquipedia(合规兜底) │
│         + HLTV 移动端 API(后期, Leclowndu93150 方案) │
├ 模型层 ─────────────────────────────────────────┤
│ 每图独立 Glicko-2/Elo + Logistic/XGBoost 差值特征  │
│ veto 序列模拟合成 BO3 概率                        │
│ 校准：Venn-ABERS（区间宽->不下注）或 isotonic       │
│ 全部特征带 as_of 时间戳，fixture 测试强制防泄漏     │ ← 0xAidan
├ 价值层 ─────────────────────────────────────────┤
│ edge = p_model − p_market − fee(0.05·p·(1−p))     │
│ p 先打 2% haircut；edge ≥5% 才出手                │
│ edge 分桶(0-5%/5-10%/…)分别统计校准与 ROI          │ ← OctagonAI
├ 下注层 ─────────────────────────────────────────┤
│ 0.25 Kelly，封顶 min(5% 资金, 盘口深度)             │
│ 闸门：身份匹配(别名YAML)/成交量/点差/深度/时间窗/样本量│
├ 执行层 ─────────────────────────────────────────┤
│ py-clob-client-v2 + asyncio 单循环 + SQLite 状态   │
│ 双 WS（market+user），maker 限价入场               │ ← poly-maker
├ 风控层 ─────────────────────────────────────────┤
│ 默认 paper；实盘需显式多重开关；日亏熔断+回撤熔断     │
│ 5 状态 regime 机（比赛开始=EVENT,拉宽报价/撤单）    │
└ 验证层 ─────────────────────────────────────────┘
│ 回测验收：Brier 技能分打赢市场 + flat-bet ROI       │
│ 独立结算对账引擎（HLTV 赛果 vs Polymarket 结算）    │
```

---

## 二、分主题要点

### A. Polymarket 机器人生态

| 项目 | 价值 | 借什么 |
|---|---|---|
| **warproxxx/poly-maker** ⭐1,458 | Python 做市标杆，CLOB V2 | 整体架构：策略=纯函数、journal 回放、5 状态 regime、doctor/livetest 链路验证 |
| **kachence/polymm** ⭐79 | 体育/电竞 MM+套利，真实盈亏（+$5k） | 血泪教训：逆向选择是主死因；对冲腿残仓是隐形亏损；死因=公共赔率源太慢 |
| **HarrierOnChain/...Toolkits** ⭐408 | Rust，10 策略 | venue 适配器抽象；circuit breaker/depth guard/trade floor；链上解码比公开 API 早 3-30s 看到大仓 |
| **0xAidan/esports-model** | 与我们目标相同 | 全部（见上） |
| warproxxx/poly_data ⭐2,284 | 链上全量订单流->CSV | CS2 市场微观结构/定价滞后研究的数据源 |
| pmxt ⭐2,093 | 预测市场版 CCXT | 多平台抽象备选 |

SDK 现状：官方 `py-clob-client-v2`（交易）+ `py-sdk`/`polymarket-client`（统一数据）。**旧 py-clob-client 于 2026-05 死亡，任何还在用它的项目/教程全部过时**。GitHub 上有一批 keyword-stuffed 仿冒仓库（SEO 诈骗），已剔除。

### B. CS2 预测 ML（可信基准）

- **可信的赛前方向准确率：65-77%**。任何宣称 85%+ 的都是回合内状态泄漏或过拟合，勿信。
- `haidamykyta/cs2`：XGBoost+isotonic，每图 Glicko-2；真实 walk-forward 成绩 = 73% 方向 / 35.85% 投注胜率 / +2.26% ROI / CLV +0.23；实盘教训沉淀成硬规则（margin>7% 不下、stand-in 地图胜率折价 30-40%、低样本地图(<15场)不可信、每日限注 2）
- `0xAidan/esports-model`：v1 刻意用 Logistic+8-10 个差值特征；防泄漏 fixture 测试；别名 YAML 四级匹配解决"NAVI vs Natus Vincere"；只做 series winner 过滤 map winner
- `Leclowndu93150/pickem-prediction-model` ⭐17（本领域最高星）：**HLTV 移动端 API 逆向**（curl_cffi TLS 伪装，67 端点：VRS、地图池 CT/T 分边、pick/ban 频率、Rating 3.0 趋势）+ Major 瑞士轮 Monte Carlo + **BO3 veto 序列模拟**（ban 对手最强图->选己方最佳->decider 逐图合成）
- `d-roho/CSGOPredictor`：GSI 回合快照 + 23 特征，强调校准>准确率；训练数据 Kaggle SkyBox 12.2 万快照
- 学术：veto 决策建模可提升 11% 地图胜率（arXiv 2021）；ESTA 数据集（含轨迹）

### C. Demo 解析（ML 规划 ②③ 层的底座）

- 基座组合：**demoparser2（提取，<1s/场）+ awpy（地图坐标/可见性/nav）+ awpy-data（地图资产每日 CI 自动跟 Valve 版本）**
- `benginN/csfreezetime`：最完整的战术情报平台参考--Ghost Rounds 轨迹叠加、15 秒默认站位快照、道具落点聚类、16Hz 采样 + 预计算特征表 + ClickHouse
- `Santtu777/CS_demo`：地名->战略区规则法分 15+ 种 T 方战术，**无需 ML，③层第一步直接抄**
- 空白点：**没有项目把 demo 轨迹特征接进盘口预测**；回合预测现有工作全基于回合快照
- 成本：解析不是瓶颈（HLTV 下载限速才是）；Tier-1 战队一年 demo 20-30GB
- Demo 获取：ReagentX/HLTVDemoDownloader + Playwright 过 Cloudflare；FACEIT 开放 API

### D. 量化框架方法论

- 技术栈共识：pandas + XGBoost/LightGBM + SQLite + APScheduler + FastAPI；无项目用 MLflow（实验跟踪存 SQLite 即可）
- 校准：Venn-ABERS（ip200/venn-abers，活跃）输出概率区间，区间宽=不下注，小样本电竞场景比 isotonic 稳健
- Kelly 实测值：weather-bot 15%、OctagonAI 50%（组合降到 25%）、haidamykyta 25%——**共识 0.25 + 三重封顶**
- 验收标准：**Brier 技能分打赢市场**（不是准确率）；edge 分桶统计；CLV 微正是健康信号
- Paper 模式：所有认真项目 paper 是默认态、live 是多重显式开关、paper/live 同一套代码路径

---

## 三、直接可抄清单（按优先级）

1. **0xAidan/esports-model**：数据源选型文档、as_of 防泄漏工程、Polymarket EV 数学（fee/haircut/Kelly/深度封顶）、别名匹配 YAML、六道闸门
2. **haidamykyta/cs2**：每图 Glicko-2、特征清单、以及它的实盘硬规则（那些规则每一条都是亏钱换来的）
3. **warproxxx/poly-maker**：执行层架构、paper/doctor 模式、regime 状态机、TOML 分层配置
4. **Leclowndu93150**：后期要 HLTV 深数据时的移动端 API 方案；veto 序列模拟
5. **Santtu777 + csfreezetime**：③层战术特征的两步走（规则法 -> 聚类/轨迹）
6. **OctagonAI**：5 道风控闸门逻辑（TS 写的，逻辑照抄）

## 四、期望管理（诚实结论）

- 无泄漏基准：方向准确率 73-77%，投注级 ROI 个位数百分比，CLV 微正
- 头部大赛市场深（EWC 单场 $400K-877K 成交），challenger 层流动性极差
- **可下注信号稀疏，"整天全 PASS 是正常结果"**——系统的正确形态是过滤器，不是下单机器
- 我们的差异化：①CS2 生态位空白；②半职业领域知识做"赔率源"（polymm 死于慢速公共赔率，我们的信息来自自己的判断和 demo 级数据）
