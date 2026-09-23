# 自动采集会话：发现 → 配对 → 三路采集 → 审计

记录日期：2026-09-19。阶段 1 第二部分。只读、纸面模式；不下单、不碰钱包。
复用三个已验证采集器（fivee_live / fivee_mqtt / market_raw_capture），不建第四套格式。

## 自动发现（两个公开源，已实测，证据在 data/fivee/discovery_probe/）

| 源 | 接口 | 说明 |
|---|---|---|
| 5E 赛程 | `GET https://esports-data.5eplaycdn.com/v1/api/csgo/matches?page=&limit=` | 与既有采集器同主机同协议；**按赛事分块非时间序**（page1=10/1-11，page2=9/26-10/1），必须翻页（默认 4 页 ×50） |
| Polymarket | `GET https://gamma-api.polymarket.com/events?active=true&closed=false&tag_slug=counter-strike-2&order=startDate` | 单场比赛事件在此 tag 下；`cs2` tag 只有长期大盘，不可用 |

边界：`www.5eplay.com` 页面在阿里云 WAF 后面（acw_sc__v2 JS 挑战）。
**不解 WAF 挑战**（与 HLTV 约束同级的红线），所以 5E 侧发现只走 5eplaycdn 公共数据接口。

## 配对闸门（全部通过才启动；任何歧义只记录、不启动）

1. 双方队名已知（5E "TBD"/待定 直接拒绝）；
2. 归一化队名集合相等：`normalize_team` = 去重音/标点/大小写 + 去通用后缀
   （esports/gaming/club/gg/team），**保留** academy/junior/fe（不同队！）；
   别名表 `--alias-file`（JSON {"别名":"规范名"}），内置 natus vincere→navi；
3. `|5E plan_ts − Polymarket startDate| ≤ --tolerance-minutes`（默认 45）；
4. **双向唯一**：一个 5E 比赛只能配一个 PM 事件，反之亦然；多方候选 → ambiguous，
   写入 `data/live_sessions/discovery_latest.json`，永不自动启动。

启动前还有第二道闸 `validate_pair_at_start`：用**新鲜** event 载荷重验
（未关闭、标题队名与发现时一致、Map 1 Winner 盘口存在）。失败写入
`data/live_sessions/start_rejections.jsonl`。

## 用法

```bash
# 干跑：只发现+配对+写报告（data/live_sessions/discovery_latest.json）
python -m cs2ml.live_session_run --mode dry-run

# 单次：发现一轮，若有「已到启动窗口且配对唯一」的比赛则采一场
python -m cs2ml.live_session_run --mode once

# 守候：每 --poll-minutes 轮询，比赛进入启动窗口即自动三路采集+审计
python -m cs2ml.live_session_run --mode watch --watch-hours 8

# 手动指定（跳过发现，但保留全部启动闸）
python -m cs2ml.live_session_run --mode once --match-id csgo_mc_xxxx --event-id 10xxxxx
```

关键参数：`--lead-minutes 20`（开赛前启动窗口）、`--max-lateness-minutes 120`、
`--hours 4`（采集时长）、`--pages 4`（5E 翻页）、`--alias-file aliases.json`。

## 会话产物（data/live_sessions/<session_id>/）

- `fivee_events.jsonl` / `fivee_states.jsonl`：5E 事件流 + MQTT 状态流（含 meta）；
- `market/`：单 WS 订阅全部 Map Winner token 的**原始深度**（events/metadata/summary）；
- `session_manifest.json`：三路采集器退出码与错误；
- `session_spec.json`：bout↔map↔market 绑定（map_name 从 5E 事件日志多数投票解析；
  解析不出 → spec 构建失败 → 审计 fail-closed 拒绝）；
- `reports/live_session_<id>/audit.json`：live_session_audit 的 frozen 判定。

## 已知限制

- **覆盖率错配**：Polymarket 的 CS2 单场多为 CCT/南美区域赛；5E 覆盖国际赛。
  双方都有的比赛才会配对（如 CCT Europe、magic–MIBR 这类）。错配窗口期
  （如 9/19-9/24）paired=0 是真实状态，不是故障。
- BO1 赛事没有 Map Winner 盘口 → 永不配对（只录 BO3 起的图级盘口）。
- 5E `state.status` 枚举未完全核实（只记录原始值，不做猜测性过滤）。
- watch 模式需进程常驻（tmux / 计划任务）；进程被杀 = 采集段无 session_end，
  审计会正确判 incomplete（异常退出永不被误报为完整）。

## 测试

`tests/test_live_auto_session.py` 24 项：归一化（含 academy/junior 区分）、
标题解析、TBD 拒绝、翻页去重、时间窗、双向唯一歧义、别名桥接、spec 构建
+ 审计 fail-closed。全套 460 项通过。
