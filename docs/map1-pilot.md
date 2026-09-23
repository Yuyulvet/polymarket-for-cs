# Map 1 预测与纸面验证：第一版

本入口实现「地图已确定、阵容已确认、尚未开打的 Map 1」：
历史数据 → 时间安全特征 → 单图胜率 → 指定 Map 1 市场 → 真实盘口成本 → 延迟纸面成交 → 显式结算。
它不使用 `predict.py` 的系列赛概率，不调用钱包或下单接口，不自动启动后台采集。

需要页面操作时，使用 [本地纸面工作台](map1-workbench.md)。开发边界见
[开发与验证约定](map1-development-plan.md)：先开发、验证，之后才单独讨论实盘。

## 快速运行

在项目根目录执行（Windows）：

```powershell
.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.venv\Scripts\python.exe -B -m cs2ml.map1 prepare
```

产物位于 `data/map1/`，不覆盖旧实验缓存：

- `history.parquet`：独立地图、固定 roster 视角的赛果与历史表现。
- `features.parquet`：Map 1 预测特征及最大历史可用时间。
- `predictions.parquet`：开发期 walk-forward、冻结留出集预测与训练截止时间。
- `validation.json`：数据排除原因、概率指标、校准分箱及局限。

依赖沿用项目环境，parquet 读写显式依赖 `pyarrow`。全部测试使用标准库 `unittest`，不访问网络。

## 数据与时间纪律

每张地图以规范化的 `demo_path` 为唯一身份，同一系列赛不同地图不会拼接回合。
地图序号从文件名末尾的 `-m1-mirage.dem` 读取，避免把队名 M80 识别成第 80 张图。
双方身份为各自五个 Steam ID 的有序集合；A/B 是稳定 roster 排序，不是当前 CT/T，
也不从市场结算价格反推队伍身份。

回合胜者从 `winner_side` 重建。当前只支持 MR12、重复 MR3，
常规半场换边，加时中场换边、连续加时之间保留上一半场阵营。
逐回合检查是否已终局，截断、重复、非连续回合、换人和不符合该赛制的录像会明确排除。
未修改旧 demo 解析器或其缓存。

历史统计只使用 `available_at < decision_at` 的完整比赛，并排除目标系列赛。
`available_at` 当前定义为关联比赛的 `end_date + 300 秒`；可通过 `--publication-lag` 调整。
这是**结果/统计发布时刻的代理假设**，不是当时实际收到完整 demo 的时间。
历史预测默认取计划开赛前 60 秒，`--lead-seconds` 可调整。

特征包含跨地图、特定地图的胜率、回合胜率、手枪局、CT、K/D、ADR、首杀和补枪差分。
使用固定收缩先验、90 天半衰期，不从全量未来数据估计先验。
精确 roster 历史不足会退回先验；前向录制时任一方少于三张历史图则阻止生成候选交易。
当前没有完成选手跨账号合并、替补阵容迁移、对手强度校正或选图方特征。

模型是标准化后的正则逻辑回归。镜像增广只发生在训练切分之后；
只有在预测前已结束且结果可用的系列赛才能训练。
最后约 20% 的预测时间组是留出集，模型在进入留出集前冻结；
期间允许更新当时已经公开的历史比赛统计，但不使用留出集结果重新拟合模型。
没有根据留出集调阈值或选模型。每条预测保留训练量、最大结果可用时间。

历史数据没有保存地图 veto 和首发名单的公告时间。因此离线指标只表示
「假设当时地图和阵容已知」的模型诊断，**不表示能按当时价格交易，不计算历史交易收益**。
不要拿赛后实际地图特征与提前几小时的价格直接配对。

## 市场检查和前向录制

只读检查公开市场，无需私钥：

```powershell
.venv\Scripts\python.exe -B -m cs2ml.map1 inspect --event 1026904
```

该 ID 是开发时的历史核验实例，不是交易建议；实际录制应使用当时尚未开始的比赛。

`record` 需要一个经确认的上下文 JSON。下面仅说明字段结构，替换示例值后才能使用：

```json
{
  "event_id": "实际Gamma事件ID",
  "map_no": 1,
  "map_name": "de_mirage",
  "scheduled_start_at": "2026-09-20T10:00:00Z",
  "map_known_at": "2026-09-20T09:50:00Z",
  "map_source": "地图公布来源URL或保存的记录路径",
  "roster_known_at": "2026-09-20T09:45:00Z",
  "roster_source": "首发名单来源URL或保存的记录路径",
  "team_a": {
    "outcome": "与Polymarket outcome一致的队名",
    "roster": ["SteamID1", "SteamID2", "SteamID3", "SteamID4", "SteamID5"]
  },
  "team_b": {
    "outcome": "另一队的Polymarket outcome",
    "roster": ["SteamID6", "SteamID7", "SteamID8", "SteamID9", "SteamID10"]
  }
}
```

上下文的公告时间和来源由采集者提供；程序检查时间顺序、来源非空和身份一致性，
不声称已经自动验证来源内容。第一版不自动从赛后数据填这些字段。
后续实时数据源应输出相同结构，并保留原始接收证据。

```powershell
.venv\Scripts\python.exe -B -m cs2ml.map1 record --context data/map1/context.json --count 20 --interval 3
```

录制有界，默认只取一条。保存完整双边订单簿、服务器时间、请求/接收时间、
事件元数据、完整结算规则、费率、撮合延迟、预测、历史覆盖和拒绝原因到 `snapshots.jsonl`。
每次读取的是最新 `feeSchedule` 与 `secondsDelay`，不写死费用或延期天数。
严格校验事件 ID、开赛时间、二元 `child_moneyline`、Map 1 标题、condition ID 和队伍 token；
没有整场盘口回退，没有模糊子串匹配，不删除 Academy 等身份标记。
缺费率、过期簿、空簿、盘口交叉、地图/阵容尚未确认或已经开赛都会阻止候选交易。

## 纸面成交与结算

```powershell
.venv\Scripts\python.exe -B -m cs2ml.map1 replay --snapshots data/map1/snapshots.jsonl
```

默认虚拟本金 1000、每笔 10 股、每股净优势门槛 0.05，均为实验参数。
双边分别读取真实 ask，按深度计算 VWAP，逐档计算 `shares × fee_rate × p × (1-p)`。
不把中间价当成交价，也不把对手 ask 简单取补数。

初次信号仅创建待成交记录，限价为该次所需最差买入档位。
必须等待市场撮合延迟加默认 0.5 秒客户端延迟后，在另一次新快照中确认足够的限价内深度。
没有及时的新簿、价格超过限价、状态失效、现金不足或参数变化都会取消。
该模拟采用全量成交或取消，不预测挂单排队，也不能证明过去一定能成交。
每场最多一次成功入场，第一版持有到结算，不做回合间止盈止损。

结算必须提供明确的两个 token 兑付值、condition ID、结果收到时间和来源。
不把末价接近 1 当成已经结算。JSONL 每行一个如下对象：

```json
{"event_id":"实际事件ID","condition_id":"实际conditionId","resolved_at":"2026-09-20T12:00:00Z","payouts":{"A队tokenID":1.0,"B队tokenID":0.0},"source":"最终结算依据URL或记录路径"}
```

```powershell
.venv\Scripts\python.exe -B -m cs2ml.map1 replay --snapshots data/map1/snapshots.jsonl --resolutions data/map1/resolutions.jsonl
```

50–50 结算填两个 `0.5`，按每股 0.50 兑付，绝非退回买入成本。
费用使用美元等值记账，尚未模拟真实账户以份额扣费的细节。
未结算持仓保留成本与占用现金，不在文件结束时虚构平仓或利润。

评估使用每场第一个通过校验的预测（包括未交易场次），对比模型与市场概率。
收集到至少 30 场当时已经结算的有效样本后，才拟合过去数据上的模型/市场组合；
三者另在同一可评估样本上比较。50–50/分数兑付计入盈亏，但不冒充二元标签进入 Brier/AUC。
样本不足时组合指标是 `n=0`，不虚构验证结果。

## 下一阶段

已新增本地工作台：自动找盘、人工确认 veto/首发来源与本地接收时间、持续公开簿采集、
明确二元结算核验和复盘。第一版使用 HTTP 轮询，不承诺低延迟；平分或冲突结算保留待核验。
优先积累独立前向样本、改善历史数据覆盖和模型质量，再评估是否需要自动 veto 数据源或盘中模型。
旧 `paper_trade.py`、`.scratch/` 策略搜索和历史 AUC 不属于这条新验证流程。

参考：[Polymarket 费用](https://docs.polymarket.com/trading/fees)、
[市场字段与状态](https://docs.polymarket.com/market-data/market-details)。
