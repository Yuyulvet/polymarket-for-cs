# 前向纸面账本（手枪局波段策略 · 用户判断版）

建立：2026-09-23。工具：`cs2ml/paper_ledger.py`。**只读、纸面、不接钱包不发单。**

## 为什么是用户判断、不是模型选股

`pistol_model_v3` 的 Phase-2 expanding-window OOF 判决（详见
`reports/pistol_model_v3/results_p2.json`）：**v2 老特征（对照）和 v3 文献新特征
（map×side 先验 / 手枪首杀率 / 补枪效率）在生产式协议下一起塌**——

| 协议 | control (v2特征) | v3_full |
|---|---|---|
| 70/30 单次切分 AUC | 0.557 | 0.542 |
| **expanding OOF AUC** | **0.485** | **0.482** |
| **OOF 中预测≥60% 的命中率** | **42.9% (n=28)** | **41.4% (n=29)** |
| OOF 顶decile 实际 | 0.523 | 0.446 |

结论：v2 的「top-decile 实际 ~65%、EV 首次转正」是**单次 favorable 切分的产物**；
换成赛前真实形态（滚动窗口、冷启动、窗内 Platt 校准），任何特征组合的手枪预测
都≈硬币甚至反指。**模型选股线关闭**；本账本测的是策略最后一个未测变量——
用户自己的读局质量，以市场为基准（CLV）。

## 冻结规则（建账时写死，记账期间不可改）

- **入场**：限价买 = 手枪前 plateau 窗口 `[t_p1−420s, −240s]` 的 **ask − 2pt**；
  仅当窗口内盘口真实触及限价才算成交（`filled`），否则记 NO_FILL 不进 P&L。
- **EXIT-W**（赢手枪）：`[t_p1−60s, +60s]` plateau 的 **bid** 卖出（第一平台期
  就抛，吃情绪弹；+13pt 是尾部不是常态）。
- **EXIT-L**（输手枪）：`[t_r2−60s, +60s]` plateau 的 **bid** 卖出——**机械版，
  不等回暖**（实测 R3 回暖率仅 44.4%，等趋势=硬币）。
- **点差如实**：买吃 ask、卖吃 bid，双边都付。
- **CLV**：`close_mid` = `[t_p1−900s, −600s]` 中位 mid；`clv_pt = close − entry`。
  持续正 CLV = edge 的最强前向证据（比 P&L 早）。
- **诚实性**：`record` 必须在手枪局结束前完成（时间戳为证）；settlement 只读
  磁带，不碰 5E 时点（市场领先 5E 60–120s，全部用市场时间 plateau 中位数）。

## 用法

```bash
python -m cs2ml.paper_ledger list                          # 看有哪些会话/图
python -m cs2ml.paper_ledger record --session <id> --bout 1 --pick <队名> [--notes "..."]
python -m cs2ml.paper_ledger settle  --session <id> --bout 1   # 赛后结算
python -m cs2ml.paper_ledger report
```

目标样本量 30 笔（有成交的）。判据：命中率须显著 > 53.4%（form 占优方实测
天花板）且 CLV 均值 > 0；两者皆负则该策略最后一条线也关闭。

## 基准线（预先钉死）

- 命中率基准 **53.4%**（n=586 实测，time-stable）；
- 每笔 EV 结构：赢腿弹幅 +6.25pt / 输腿跌幅 −11.2pt（磁带实测均值，含点差前）；
  盈亏平衡命中率 = 11.2/(6.25+11.2) = **64.2%**——这是每笔交易的真正及格线。
