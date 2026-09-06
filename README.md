# Polymarket CS2 交易研究

CS2（Counter-Strike 2）电竞的**胜率预测 + Polymarket 交易**研究代码库。目标：从比赛数据里提炼比市场更准的「读局」信号，并验证能否在 Polymarket 的 CS2 盘口上货币化。

> ⚠️ **诚实结论（截至 2026-09）**：统计/模型层目前**没有找到可货币化的 edge**。Polymarket 对 CS2 盘口定价有效——比分几乎是充分统计量，经济/装备信号被比分吸收，赛前模型跑不赢市场价。唯一尚未证伪的线索是 **latency edge（先于市场读到局面变化）**，这需要实时观赛线，是当前工作重心。

---

## 项目分层

项目分三层，从「结果」到「过程」到「实时」：

### 1. 结果级预测（match-level）

只用赛果（谁赢谁输），做 as-of 防泄漏的赛前胜率预测。

```
bo3.gg API ──> backfill.py ──> data/cs2.db (SQLite)
                                    │
                             features.py（as-of 防泄漏特征）
                             ├─ Glicko-2 评级（ratings.py，每日 RD 衰减）
                             ├─ 近期状态（5/10/20 场窗口，90 天内）
                             ├─ H2H（2 年窗口，0.5 先验）
                             └─ 经验/休息/队龄/赛事级别/奖金
                                    │
                             train.py（expanding-window walk-forward）
                             ├─ Logistic（标准化）
                             └─ LightGBM + isotonic 校准（窗口内 CV）
                                    │
                             evaluate.py ──> reports/eval_report.md
                             predict.py   ──> 未来比赛胜率 CSV
```

**防泄漏纪律（改动前必读）**：

1. 每场特征只允许用 `start_date` 严格早于它的比赛（features.py 时序游走保证）
2. 评级更新发生在特征产出**之后**（`apply_result` 在 `compute_features` 之后）
3. walk-forward：训练窗口 < T，测试窗口 [T, T+30d)，绝不重叠
4. 镜像增广只用于训练集
5. bo3.gg 的 `ai_predictions` 是其赛前预测，只作 baseline，不作特征
6. 任何新特征必须能通过「as-of 时间戳」测试

**关键字段**：`match`(id/slug/status/bo_type/winner_team_id/tier/start_date/team1_2_score/stars)；`upcoming` 额外带 `bet_updates`（20 个盘口赔率）、`ai_predictions`、`games`。注意 `included` 是 dict 不是 list，`ai_predictions` 可能是 dict/str/None。

- **结论**：市场比模型更准（Brier 0.2264 < 0.2422，spearman 分歧-结果 −0.26），无 edge

### 2. Demo 过程级学习（process-level）

从 HLTV GOTV demo 逐 tick 解析「**怎么打的**」，而不是「打得多好」——对应信息优势「读局」：

| 粒度 | 模块 | 产出 |
|---|---|---|
| 交火级 | `duel.py` / `engagements.py` | per-player 权重、per-player × per-place 胜率、信息差（谁在看谁）、闪致 / 补枪 |
| 回合级 | `rounds.py` / `round_transition.py` | 开局 15s 阵型、contact 打点识别、6 档装备分层、经济决策 → 下回合胜方 |
| 选手 | `player_behavior.py` / `player_style.py` | 站位 / 侵略性 / 道具 / 武器 / 买枪优先级操作画像 |
| 队伍 | `team_strategy.py` / `team_style.py` | 87 支真实队伍细粒度买枪签名、队伍战术身份 |

```
S 级赛事（bo3.gg，近 3 个月）──> collect.py ──> 下载 HLTV .rar ──> 解压 .dem
        │                                          │（下载时定队名映射 -> demos manifest）
        │                                          ▼
        │                               demo_features.py（队级聚合）
        │                               rounds.py（开局 15s 阵型 + contact 打点）
        │                               engagements.py（交火级位置 + 阵型 + 信息差）
        │                                          │
        └──────────────────────────────────────────▼
                                    duel.py（学习引擎）
                                    ├─ per-player 权重（收缩 + L2-logistic）
                                    ├─ per-player × per-place 胜率
                                    └─ P(win duel) 可预测性（分组 CV AUC）
                                           │
                                           ▼
                                    rosters.py（per-player skill -> roster -> 地图胜者）
```

**关键设计决策**：

- **位置 token = nav mesh 命名区域**（demoparser2 的 `last_place_name`）：自动编码墙/房间结构，无需下载 nav 文件
- **队伍身份用 roster**（5 个 steamid 排序元组），不用 `team_number`（BO3 跨图翻转主/客）
- **队名映射在下载时定**：collect.py 用「队名对 + 日期」把 bo3.gg 比赛映射到 HLTV match
- **交火级对称标签**：每次击杀产两个样本（killer 胜 / victim 负），特征取 kill 前 1 秒快照（无标签泄漏）
- **per-player 权重是学出来的**：`player_id` 作分类特征进 GBDT，另用贝叶斯收缩估计「选手 × 位置胜率」，不手标角色
- **信息差（info asymmetry）**：用 tick 快照 `yaw` 判断「谁在看谁」（45° 锥内）、`flash_duration` 判断「谁被闪」——看得见对方的一方胜率更高
- **道具覆盖**：交火级 `inventory` 判持有；队级 `assistedflash`（闪致击杀占比）/ `trade`（5s 内补枪）度道具有效性
- **roster skill 回灌**：5 人 skill 均值预测地图胜者，in-sample AUC 0.6（信号存在）；无泄漏 LOO 可验证性取决于同队多场 demo

### 3. 实时交易线（进行中）

把「读局」接到实时市场，纸面交易验证（**不碰真钱**）：

```
bo3.gg live（地图比分 + 庄家赔率） ──> live_state.py
Polymarket CLOB WebSocket 秒级价    ──> realtime_record.py
预测 → 买入 → 实时判断 → 卖出        ──> paper_trade.py（纸面状态机）
```

---

## 目录结构

```
cs2ml/                  # 主包（全部模块）
  backfill.py           # bo3.gg 历史回填 -> data/cs2.db
  features.py           # as-of 防泄漏特征
  train.py / evaluate.py / predict.py   # 结果级预测流水线
  collect.py / hltv.py  # 下载 HLTV GOTV demo（.rar -> .dem）
  demo_parse.py / demo_features.py / rounds.py / engagements.py  # demo 解析
  duel.py / rosters.py  # 交火级 + roster 级学习
  round_transition.py   # 回合级经济决策模型
  player_behavior.py / player_style.py / team_strategy.py  # 操作画像 / 队伍策略
  live_state.py         # bo3.gg live 轮询（地图比分 + 赔率）
  realtime_record.py    # Polymarket CLOB WebSocket 录价
  paper_trade.py        # P4 纸面交易状态机
docs/                   # 设计 / 调研文档
reports/                # 评估产出（eval_report.md 等）
scripts/                # 探测 / 冒烟脚本
data/                   # 生成产物（SQLite / parquet / demo，gitignore）
models/                 # 预留（模型内联训练，不落盘）
```

---

## 环境与安装

```powershell
# 1. 创建虚拟环境
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install -r requirements.txt
```

依赖（`requirements.txt`）：`requests`、`pandas`、`scikit-learn`、`lightgbm`、`joblib`、`curl_cffi`、`rarfile`、`demoparser2==0.42.0`。

> 需在 Windows 上运行（demo 解压依赖系统 `7z` / `rar`）。`config.py` 里的路径默认相对于项目根目录，无需额外配置；bo3.gg 与 Polymarket 均无需鉴权 token。

---

## 使用

### 结果级预测

```powershell
# 历史回填（可断点续传）
python -m cs2ml.backfill --start 2024-01-01 --end 2026-08-23

# 训练 + 无泄漏评估
python -m cs2ml.train

# 评估报告
python -m cs2ml.evaluate

# 预测未来比赛
python -m cs2ml.predict --date 2026-08-29 --days 3
```

### Demo 过程级学习

```powershell
# 采集 S 级赛事 demo（每场 ~540MB，先 dry-run 校验队名映射）
python -m cs2ml.collect --dry-run
python -m cs2ml.collect --months 3

# 交火级学习引擎（per-player 权重 + 位置胜率）
python -m cs2ml.duel

# 回合级阵型 / 打点识别
python -m cs2ml.rounds

# roster 级 skill 回灌
python -m cs2ml.rosters
```

### 实时交易线（纸面）

```powershell
# 列出当前 live 比赛 + 匹配到的 Polymarket 盘口
python -m cs2ml.paper_trade --list

# 对某场 live 比赛跑纸面交易（虚拟买入 -> 判断 -> 卖出，全量落盘）
python -m cs2ml.paper_trade --team1 Spirit --team2 MOUZ --minutes 60
# 日志 -> data/paper/paper_*.jsonl
```

---

## 数据源

| 源 | 用途 | 说明 |
|---|---|---|
| **bo3.gg** | 赛程 / 赛果 / 赔率 / live 地图比分 | `/api/v2/matches/*`，无需 token |
| **HLTV GOTV** | demo 文件（.rar） | 赛后 ~90s 延迟，逐 tick 解析过程数据 |
| **Polymarket** | CLOB（下单 / 价）/ Gamma（盘口发现） | `clob.polymarket.com`、`gamma-api.polymarket.com` |
| **CS2 GSI**（规划） | 回合级经济 / 击杀 / 存活（实时） | Valve 官方，自托管 GOTV 观战，唯一免费全字段路径 |

回合级实时游戏内数据（经济 / 击杀 / 存活）当前**没有免费托管源**：HLTV scorebot 反爬混淆、bo3.gg live 只有地图比分 + 赔率。可选路径见 `docs/data-sources.md` 与 `docs/cs2-markets-guide.md`。

---

## 关键结论（诚实盘点）

历史上多数方向是**负面结果**，这里如实记录，避免重复踩坑：

- **赛前市场回测**：模型 Brier 0.2422 vs 市场 0.2264（市场更准），spearman 分歧−结果 −0.26（反向）→ **无 edge**
- **盘中经济信号**：EV +0.017 ≈ 「押比分领先方」动量（已定价），非 alpha
- **30s 中局模型**：存活差 + 首杀把回合 AUC 0.75 → 0.80（读局有效），但这是 **latency edge**，聚合到分钟级盘口无增量
- **回合经济决策**：经济 → 下回合胜方 AUC 0.745（真实信号），但被比分吸收（0.82 → 0.95）
- **per-player form**：跨图 K/D → 赛前 map winner AUC 0.62（时间序诚实），超 roster 强度 0.600，但跑不赢市场价
- **仓位管理**：负 edge 无法靠先买 / 补仓 / 止损弥补

**核心矛盾**：模型能「读懂」比赛（回合级 AUC 0.75~0.80），但市场已经把比分 + 经济都定价了；要货币化只剩「**比市场更早读到**」的 latency edge，这正是实时观赛线（第 3 层）在做的事。

---

## 为什么 `models/` 是空的

模型在训练时**内联构建、不序列化**：`LogisticRegression`、`LGBMClassifier`、`HistGradientBoosting`、`KMeans` + `StandardScaler` / `OneHotEncoder` / `Pipeline` / `ColumnTransformer`，用 `GroupKFold(match_id)` 交叉验证 + `cross_val_predict` 即时产出预测。历史实验多、结论多为负面，所以没有落盘一个「最终模型」——需要时由 `train.py` / `predict.py` 现场训练。

---

## 已知陷阱（改代码前必读）

1. **team identity**：demo 的 `home_win` 是「上半场 CT 赢没赢」，**不是** team1 赢。BO3 换边使 ~31% Map1 / ~65% Map2 被反。正确口径：队伍身份用 `roster_key`（5 个 steamid 排序拼接），胜负用 raw 结算（token 末价 ≥0.9）。
2. **steamid 分裂**：同一选手跨 demo 可能多个 steamid（donk 89+4），per-player 建模须按 name 合并。
3. **Map 2+ 不可对齐**：盘中价逐图结算漂移，回测只 Map 1 切片可信。
4. **demo 截断**：GOTV 有时截断（胜方不足 13 分），`demo_features` 用 `complete` 标志兜底。
5. **roster → team_id 对齐未做**：把 roster skill 注入 match 级特征需知道「哪个 roster = team1/team2」，需 HLTV lineup + 多场同队 demo。
6. **磁盘**：全量 S 级近 3 个月 ≈ 169 场 × 540MB ≈ 90GB，解析后按需删 .rar。
7. **武器泄漏**：`player_death` 只带击杀者武器，对 victim 样本是标签泄漏——须从 tick 快照取双方各自 `active_weapon_name` 再归到武器家族（已修，改代码别回退）。

更细的设计见 `docs/`（`data-sources.md`、`cs2-markets-guide.md`、`strategy-analysis.md`）。

---

## 免责声明

本项目纯属**研究与实验**，不构成任何投资建议。所有交易回测与纸面交易均为模拟；历史性能不代表未来收益。预测市场存在本金损失风险。
