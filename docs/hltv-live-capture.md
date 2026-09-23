# HLTV 实时原始采集原型

2026-09-16：用户要求继续开发公开采集。已实现独立、短时、只读原型，**没有接入模型、纸面下单或实盘**。
HLTV 条款限制抓取/数据挖掘，用户选择继续研究不等于取得平台许可；本实现不处理验证码、登录、
指纹伪装、代理轮换或访问限制绕过。一次失败即停止，不切换端点或协议重试。

## 本次真实验证

- 浏览器中公开比赛页可见 Scoreboard、Game log 和击杀条目。
- 普通 HTTP GET 成功，读取比赛页 `#scoreboardElement` 的公开属性。
- 比赛：SPARTA–BAKS，HLTV ID `2397671`；页面提供 `https://scorebot-lb.hltv.org`。
- 随后一次 Engine.IO 3 / WebSocket 握手返回 **HTTP 403**，立即停止。
- 收到的实时事件数量为 **0**。没有验证当前服务协议兼容性、字段覆盖、源事件时间或延迟。
- 证据目录：`data/hltv_live/probe-20260916-sparta-baks/`，包括页面、发现元数据和停止原因。
- 完整离线回归 297 项通过；其中本轮新增 58 项（日志 17、协议 21、采集编排 20）。
  测试证明所覆盖的代码行为，不证明服务可访问、字段实时或策略盈利。

页面可见不代表脚本事件连接已获接受；403 的具体成因未确定，不把它断言为某一种反爬、权限或协议问题。
浏览器中已有击杀条目也可能包含历史回放，不能据此认定为新鲜事件。本轮不再尝试其他连接方式。

## 用法与边界

在明确允许访问且无需绕过访问限制的环境中，可手动进行一次短时验证：

```powershell
.venv\Scripts\python.exe -B -m cs2ml.hltv_live --match-url "https://www.hltv.org/matches/<id>/<slug>" --seconds 30
```

默认创建新的 `data/hltv_live/<UTC>/` 目录；指定已存在目录会拒绝运行，不覆盖旧证据。
只读取一次比赛页，只连接它公布的已审查端点，无后台常驻、自动重连或市场交易操作。
默认录制 30 秒，允许 1–300 秒；这是事件连接阶段时长，另有页面读取、握手及关闭超时。
默认日志上限 10,000 条、16 MiB，先到上限即停。日志预算包括 `metadata.json` 和 `events.jsonl`；
另有小型 `summary.json`，便于预算耗尽后查看停止原因。

入口：`cs2ml/hltv_live.py`；纯协议状态机：`cs2ml/hltv_scorebot_protocol.py`；
日志：`cs2ml/live_feed_store.py`。新增依赖 `websocket-client>=1.8,<2`，当前虚拟环境已有该库。

### 日志语义

- `socket_packet` 原样保留收到的文本消息；顺序号逐次递增，重复到达不去重。
- `received_at` 与 `monotonic_ns` 在记录器入口采样，指应用接收后的记录时间，**不是游戏发生时间或网卡到达时间**。
- `scorebot_event` 是解码派生记录；其 `packet_received_at`、`packet_monotonic_ns` 和 `raw_sequence`
  指回原消息，不拿派生记录时间冒充接收时间。
- `source_timestamp` 保持 null；没有发现并核实源时间前不猜测、不回填。
- `log/fullLog` 内层字符串不重写。`fullLog` 标记可能回放，其他消息也不保证新鲜。
- 所有事件的 `eligible_for_inference` 固定为 false。连接成功、订阅已发或收到事件均不代表已建立可信游戏状态。
- 二进制/无效 UTF-8 消息以 base64 留证后停止；超限消息只记大小，不保存超限正文。
- 每次写入 flush，但不 fsync；进程/系统故障仍可能丢失尾部数据。I/O 失败关闭日志，不静默恢复。
- 1 MiB 消息限制在 WebSocket 库组装消息之后检查，是应用接收/保存限制，**不是底层帧缓冲内存硬上限**。

## 尚待完成

需要可正常使用的事件连接才能检验：初始全量状态、增量与回放、换边/切图/暂停/重赛、乱序与缺失、
玩家 ID 和真实可用字段，以及与同期完整盘口的接收时间对照。本原型不提供这些已验证的保证。
尤其不能将 `money` 当装备价值、将击杀坐标当所有玩家持续位置，或把历史 demo 全知特征直接喂给 live 模型。

协议实现参考历史非官方库（不是官方 API 或当前可用性承诺）：
[HLTV 库作者源码](https://github.com/gigobyte/HLTV/blob/master/src/endpoints/connectToScorebot.ts)、
[Engine.IO v3 协议](https://github.com/socketio/engine.io-protocol/tree/v3)。
来源约束：[HLTV 条款](https://www.hltv.org/terms)。
