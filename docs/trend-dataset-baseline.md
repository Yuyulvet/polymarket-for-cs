# 阶段 2/3：统一趋势数据集 + 盘口基准模型

记录日期：2026-09-19。只读、纸面模式。协议：`cs2_swing_trend_v1`
（docs/trend-swing-protocol-v1.md）。

## 阶段 2：`cs2ml/trend_dataset.py`

把一个已审计采集会话的三路原始记录（5E 事件流 + 5E MQTT 状态流 +
Polymarket 原始盘口深度）转成严格时间安全的决策点长表
（一行 = 一个触发 × 一个图 × 一个结果 token）。

**触发点（只有机制转换，禁止逐秒/逐报价）：**

- `post_pistol_freeze`：5E 事件 type 1 且 `round_num == 2`（仅 decision_eligible 行）；
- `round_end_economy_update`：5E 事件 type 2（回合结束比分和 = 回合号）；
- `prematch_prior_dislocation`：会话首个可成交盘口簿。

**时间链（协议冻结）：**

```
info_mono   = 触发信息本地收到时间（同进程单调钟）
t_exec      = info_mono + MODEL_LATENCY_SECONDS(0.0 占位) + seconds_delay(CLOB, 默认 1)
entry       = t_exec 之后首个 best_ask（绝不成交更早的报价）
exit(h)     = t_exec+h 之后首个 best_ bid，h ∈ {15,30,60,120}
net(h)      = exit_bid − entry_ask − 双边 taker 费
```

**Fail-closed 规则：**

- initial/recovery 的 5E 事件和状态快照既不触发也不参与 join；
- 状态 join 只向后看（recv_mono ≤ info_mono）；回合胜负只来自当时已收到的
  round_end 事件（触发自身的 round_end 在 info_mono 时刻已知，允许计入）；
- 费率/延迟缺省时经可注入的 `fetch_clob` 从 CLOB 公共接口补查
  （`clob.polymarket.com/markets/{conditionId}` → taker_base_fee / seconds_delay），
  报告记录 `param_source`；
- 标签缺失保持 NaN/False，永不回填；`market_format` 非
  `read_only_raw_evidence_multi`（如旧 normalized_top 采集）整段拒绝，
  因为与 5E recv_mono 不同钟不可拼接；
- 每行带 `game_*` 状态列（比分/存活/HP/现金/手枪胜者/连败），
  阶段 3 不使用（阶段 4+ 测增量）。

**产物**：`rows.parquet` + `build_report.json`（触发统计、丢弃原因、
协议哈希、四路输入 SHA-256）。

```bash
python -m cs2ml.trend_dataset --session-dir data/live_sessions/<id> \
    --spec data/live_sessions/<id>/session_spec.json \
    --root . --clob-fees --output reports/trend_dataset_<id>/
```

**验收**：`tests/test_trend_dataset.py` 16 项，含「抽查一行能从原始记录
手工复现」（SpotCheckTests）。magic–MIBR 尾段测试：旧 normalized_top
采集被正确拒绝（0 行，`market_format_not_time_joinable`），报告在
`reports/trend_dataset_magic_mibr_tailtest/`。

## 阶段 3：`cs2ml/trend_market_baseline.py`

只用阶段 2 行的**盘口列**（`MARKET_FEATURES` 15 个：book_bid/ask/spread、
mid_at_entry、other_mid、5/15/30s _mid 收益、60s 波动、双边深度、
深度失衡、60s 成交流量/笔数、跳价旗标）。游戏状态列按构造排除。

**模型**（每个目标四个同样本比较）：constant（基线率/中位数）、
random_walk（当前价即概率 / exit=当前 bid）、L2 逻辑回归（标准化管线）、
XGBoost（3.4.1，浅树+子采样）。

**评估**：整会话前向——按会话首个决策时间排序，扩张窗训练、下一会话测试，
无序列交叉；<2 会话只给 `insufficient_sessions` 诊断不出分数。
指标：Brier、logloss、10 桶校准、bid MAE、净收益 mean/median/q05
（概率指标不替代净收益指标，协议规则）。

```bash
python -m cs2ml.trend_market_baseline --rows reports/trend_dataset_*/rows.parquet \
    --output reports/trend_baseline_v1/
```

**验收**：`tests/test_trend_market_baseline.py` 10 项（单会话拒绝、
前向切分归属、四模型×四 horizon 全部出分、常数分类器 Brier ≤ 0.25、
聚合一致性）。全套 486 项通过。

**待开展**：真实合格会话积累（watch 模式 9/25+ 窗口自动采集）后跑首个
真实前向基线；阶段 4 在相同样本上叠加 game 特征测增量。

## 阶段 4：`cs2ml/trend_game_increment.py`

问题：游戏状态信息在**相同样本**上能否超过盘口基线？三个同样本版本：

- `market_only`：阶段 3 的 15 个盘口特征；
- `game_only`：20 个编码游戏特征（比分/回合/换边码 ±1/存活差/HP差/
  现金差与份额/手枪胜者码/连败/下包/`game_state_present` 缺失指示）；
- `market_plus_game`：并集。

机制切片：**4A** 手枪经济（post_pistol_freeze 行）、**4B** 经济断裂
（round_end_economy_update 行）、**4C** 赛前错位（prematch 行——无实时
游戏状态属构造使然；赛前 roster 先验尚未进数据集，报告标注
`no_prematch_prior_feature_yet`，不进 delta 结论）。

Delta = 候选 − market_only（Brier/logloss/bid_mae/net_mean；负为改善）。
评估器仍是整会话前向扩张窗；退化切片（如单类别训练集）记录
`skipped: fit_error:*` 不中断报告。样本量小期间一切数字只是诊断。

```bash
python -m cs2ml.trend_game_increment --rows reports/trend_dataset_*/rows.parquet \
    --output reports/trend_mechanism_v1/
```

**验收**：`tests/test_trend_game_increment.py` 13 项（编码/缺失指示/
特征集不交/切片划分/同样本 n_test 相等/delta 形状/4C 标注/单会话拒绝/
random_walk 无价列回退）。全套 499 项通过。

## 阶段 5：`cs2ml/trend_swing_engine.py`

波段引擎：冻结的入场/退出/风险规则 + 确定性纸面回放。引擎不发明成交——
每行自带协议可执行报价（entry=之后首个 ask、exit=之后首个 bid、双边费已扣），
引擎只决定**是否**开纸面仓并执行风险上限。无中间价、无窗口最高价退出、
无回填（协议禁止项在阶段 2 数据里是结构性的）。

**入场闸（全过才开）**：机制白名单 → 该 horizon 标签有效（无退出报价即
跳过，fail-closed）→ `p_fair − entry_ask ≥ min_edge` → `entry_ask ≤
max_entry_ask`（冷门上限）→ `book_spread ≤ max_spread` → 每 (会话,图)
独占 + 会话开仓数上限。**退出永远是固定 horizon 的可执行 bid**——止损/
市价退出是阶段 6+ 的独立候选方案。

**p_fair 可插拔**：`market_prior`（无 edge 对照：fair=mid，有价差时
数学上永远过不了 min_edge，必须零成交）；`game_heuristic`（冻结占位
公平值：比分差/经济份额/手枪结果的 sigmoid，按 token 所属队取向，
等阶段 4+ 模型供给校准概率后替换）。

**报告**：成交/跳过原因计数、net total/mean/median/胜率、95% CI 下界
（正态近似，仅诊断——真实验收在阶段 7 冻结窗口）、分机制汇总、
rule_hash、模式标注 `paper_replay_only_no_orders`。

```bash
python -m cs2ml.trend_swing_engine --rows reports/trend_dataset_*/rows.parquet \
    --output reports/swing_v1/ --horizon 60 --min-edge 0.03 \
    --fair-source market_prior   # 对照:必须零成交
```

**验收**：`tests/test_trend_swing_engine.py` 18 项（规则校验/hash、
bout 独占、session 上限、edge/价格/价差闸、horizon 标签一致、现金账、
CI、确定性、market_prior 零成交对照、game_heuristic 对称性）。
全套 517 项通过。

## 阶段 6：`cs2ml/trend_scheme_freeze.py`

候选方案只在**预先注册**的维度上变化：fair_source ∈
{market_prior 无edge对照, game_heuristic 占位, wf_ridge_exit,
wf_xgboost_exit（walk-forward 预测退出价作公平值=「未来可成交价格模型」）}
× horizon × min_edge 等规则参数。wf_* 来源用整会话扩张窗生成
**样本外**预测：首个会话永远无预测（NaN→引擎拒成交），训练只用
标签有效行。

**选择标准（事先冻结，不许看完结果再挑）**：n_trades ≥
min_trades_floor 的候选里 net_mean 最大；平手 net_median →
net_q05 → scheme_hash（确定性）。

**冻结与防篡改**：胜出方案写入 frozen_scheme.json（含
frozen_at_utc、协议哈希、全部候选成绩、选择标准、payload SHA-256）。
`load_frozen` 重算哈希，任何字段被改都会抛
`frozen_scheme_tampered`——阶段 7 前向验收只能 `run_frozen`，
改规则=换哈希=必须带新数据重新走冻结流程，验收期间不得回改。

```bash
python -m cs2ml.trend_scheme_freeze --rows reports/trend_dataset_*/rows.parquet \
    --output reports/scheme_v1/ --min-trades-floor 5 \
    --freeze-name reports/scheme_v1/frozen_scheme.json
```

**验收**：`tests/test_trend_scheme_freeze.py` 11 项（非法来源拒绝、
hash 稳定性、首会话无预测、floor 排除、无胜者不可冻结、篡改检测、
run_frozen 与直跑胜者逐行一致、wf 端到端）。全套 528 项通过。

## 阶段 7：`cs2ml/trend_acceptance.py`

冻结方案在新比赛上的前向纸面验收 runner。判定（全部满足才 PASS）：
`n_map_units ≥ min_maps`（默认 30）且 `n_triggers_evaluated ≥
min_triggers`（默认 100）且**合并 net 的 95% CI 下界 > 0**。

硬规则：
- 账本 append-only + 幂等：session_id 已存在即跳过，重启/重复跑不重复计数；
- 每条记录冻结方案 sha256；frozen 文件变了（新冻结轮）必须显式
  `--new-round`——旧账本**归档不删除**，重新起一轮；
- `wf_*` 冻结方案必须带 `--train-rows`（冻结前时期数据）：退出价预测器
  在冻结期数据上拟合一次、向前应用，**绝不在验收数据上重训**；
- 判定每次从完整账本重算，没有会漂移的存储结论。

```bash
python -m cs2ml.trend_acceptance --frozen reports/scheme_v1/frozen_scheme.json \
    --ledger reports/acceptance_v1/ledger.jsonl \
    --session-dir data/live_sessions/<新会话> [...] \
    --train-rows reports/trend_dataset_<冻结期>/rows.parquet \
    --min-maps 30 --min-triggers 100
```

注意：CLI 路径不支持 game_heuristic 的 outcome→队映射，该类冻结方案
走 Python API（process_session 的 outcome_is_team1 回调）。

**验收**：`tests/test_trend_acceptance.py` 7 项（条目字段、wf 必须带
训练行、追加幂等、判定三态 progression、CI 为负判 failed、冻结变更
必须 new_round 且旧账本归档）。全套 535 项通过。

至此 0→7 全链闭环，唯一输入是真实配对比赛（watch 常驻即可）。
