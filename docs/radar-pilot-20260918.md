# MOUZ–NAVI 高清小地图采集原型

2026-09-18。仅只读研究；没有订单、钱包授权或自动交易。

## 本场绑定与已完成试采

- 用户提供的 5E 页：<https://event.5eplay.com/csgo/matches/csgo_mc_2398099>。
- 用户提供的 Polymarket 页：<https://polymarket.com/zh/esports/cs2/starladder-starseries/cs2-mouz-navi-2026-09-18>。
- 公开 API 核实 event 1038821；Map 1 Winner market 4641780；双方为 MOUZ / Natus Vincere，5E 使用 NAVI 别名。
- 5E 本场地图顺序为 Cache、Inferno、Mirage。试采只绑定 Map 1 Cache，不将后续地图混入。
- 直播页 <https://kick.com/starladder>，读取到 1920×1080 / 60fps。首轮参考帧可见 MOUZ 在左、NAVI 在右，Cache，比分 3–1。该单帧检查不证明整段均无回放；机器元数据继续保持 pending_manual_confirmation。
- pilot1：1,500 张小地图采样帧；5E 11 个状态快照。
- pilot2：600 张小地图采样帧；5E 5 个状态快照；Map 1 盘口 2,008 条原始 WebSocket 消息（其中可含心跳或数组快照，不等于 2,008 个价格变化）。
- 所有上述试采均已正常限时或限帧退出，并非整场持续录制。

数据位置（项目根目录下）：

- `data/stream_radar/mouz_navi_cache_20260918_pilot1/`
- `data/stream_radar/mouz_navi_cache_20260918_pilot2/`
- `data/fivee/mouz_navi_20260918_pilot1/`
- `data/fivee/mouz_navi_20260918_pilot2/`
- `data/realtime/mouz_navi_map1_20260918_pilot2/`

pilot2 盘口接收约北京时间 18:15:38–18:17:38；视频采样约 18:15:52–18:17:46；5E 会话约 18:16:05 开始。三路有重叠，但尚未人工逐事件核对，不能宣称事件时间完全同步或存在可交易延迟优势。

5E 快照包含初始 HTTP 和重连恢复状态，使用时必须按已有 decision_eligible / timing_quality 字段过滤。资金余额、单个显示武器不等于完整装备价值。此次未修改 5E 采集器；目录显示零字节曾滞后，直接读取确认已写入。

## 新增模块与安全边界

- `cs2ml/map_space.py`：七图版本化坐标元数据目录、固定地标仿射标定、独立检查点误差验证；来源锁定 awpy-data release 2000908。地图版本是否与本场一致仍需核验。
- `cs2ml/stream_radar_capture.py`：高清帧按 PTS 采样，小地图 PNG 加间隔全帧 JPG；帧数、时长、图片字节数和 STOP 文件限制；拒绝低清输入与分辨率改变。
- `cs2ml/market_raw_capture.py`：精确 event/market 绑定，保存市场规则与双方 token；保留初始数组盘口和增量原文；连接超时、时长、体积、STOP 限制。没有交易入口。源断开时退出，不假装连续。
- `data/map_assets/2000908/`：manifest、map_data 和带来源校验值的 catalog。只是元数据，不是已验证导航网格或遮挡几何。

视频 decoded_at 是本地解码完成时间，不是网络首次收到时间，更不是游戏事件发生时间；源时间未知保持 null。视频 PTS 不能直接与另一场流或市场绝对时间相减。5E source_version 时间差也不能直接等同游戏直播延迟。

全部画面 scene_status 为未核验，player_observations 为 null，eligible_for_inference 为 false。没有输出伪造站位或轨迹。Nuke 等多层地图不能从 XY 猜 Z。

## 验证与下一步

已运行且通过 20 项测试：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_market_raw_capture tests.test_map_space_radar tests.test_stream_score_verify tests.test_fivee_mqtt
```

1. 从本次采样选择正常局面、交火重叠、回放各类样例；确定真实图标可辨范围。
2. 用固定地图地标完成本场 Cache 标定，并用未参与拟合的地标复核；实际标定尚未完成。
3. 建立区域人数及短轨迹识别，记录漏检、身份交换、回放拒绝与置信度。已有校准工具不是识别器。
4. 把识别叠加回帧，与 5E 同轮状态和盘口逐事件核验，再开展战术特征和前向模型对照。

没有完成 XGBoost / 神经网络新实验，也没有得到盈利结论。

---

## 2026-09-18 第二轮:资源落地、坐标公式验证与 ROI 问题

### 已完成

- 从 awpy-data release 2000908 下载 images.zip,sha256 与 manifest 校验一致(5a1ad984...),解压出 20 张官方雷达图(1024x1024 RGBA),含 de_cache 等全部七图。注意:官方图为精绘风格,与转播简化风格差异大,整图模板匹配不可行,校准需走地标/轮廓点路线。
- **世界坐标->雷达像素公式已用数据实证确认**:`u = (world_x - pos_x) / scale`,`v = (pos_y - world_y) / scale`。用昨日 MOUZ-NRG Cache 图的 135 个 5E 击杀坐标做ground-truth,该公式下 134/135 落在官方雷达可行走区域(其余 31 个符号/轴交换变体全部不达标)。catalog.json 中 de_cache 的 pos_x=-2000, pos_y=3250, scale=5.5 经此验证。
- 未下载 geometry.zip / navs.zip(约 57MB),遮挡与导航验证留待需要时再做。

### 发现的问题(阻塞项)

- **两轮试采的静态 ROI(0,0.055,0.23,0.39 = 左上角 441x421)均未拍到雷达**。逐帧 ASCII/色块分析显示该区域内是比赛画面、选手卡片面板、队伍横幅等元素;在多个采样帧(0000001/0000101/0000151/0000300,pilot1 0000750)中均未发现雷达方块或选手点位。
- 此前"参考帧可见 MOUZ 在左、NAVI 在右,比分 3-1"应来自顶中部记分牌而非雷达;雷达在本场转播中的实际位置(或是否存在常驻雷达)待用户人工确认。
- 连带影响:地标校准、人数/轨迹识别全部依赖重新采对区域;既有 2,100 帧小地图采样帧对雷达识别无效,但全帧 JPG 仍可复用于确认雷达位置。

### 下一步

1. 用户在全帧(JPG)中指认雷达位置后,按新 ROI 重采一小段并走校准流程。
2. 校准路线修订:不用整图模板匹配,改为官方轮廓掩码 -> 广播帧点位包络/地标仿射。
3. geometry/navs 下载与遮挡验证延后,不阻塞主流程。
