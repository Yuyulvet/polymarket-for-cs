# 赛中实验修正与重建

日期：2026-09-16。范围是离线研究修正；没有部署赛中交易、接钱包、发送真实订单或接通新的 live 源。
最终完整测试：239 项通过。本轮新增 80 项回归测试，并对地图 checkpoint 路径做了独立代码检查。

## 已修正的代码路径

- `midround.py`：30 秒首杀只读取截止 tick 之前/当 tick 的敌方击杀，状态来自真实 tick 的队伍与存活快照；
  回合必须仍未结束。新增时间边界与缓存版本检查，不再静默读取旧 `midround.parquet`。
  该轻量 30 秒版本没有炸弹状态，保守排除任一方零存活，包括可能仍有悬念的下包后 T 全灭；
  有炸弹状态的事件级版本单独处理这一情况。
- `inround.py` / `inround_model.py`：明确事件 tick 与下一完整 tick 的观测时刻，隔离终局、回合外与未知时间语义；
  已下包后 T 全灭仍可能由炸弹获胜，不能一律作为终局丢掉。
- `forward_sim.py`：按单图拼回合、按稳定 roster 修正旧加时错侧，支持 MR12/MR3 和重复加时；
  当前已观测装备不被半场重置覆盖，未来经济重置从训练集估计，缺少转移/概率时拒绝预测，不默认为 50%。
- `live_map_model.py`：不再排除全部加时地图；只生成仍有下一回合的 checkpoint。
  比分与已完成回合结构按同一批地图比较，训练只用在测试系列赛开始前已可用的结果，标准化也仅拟合训练集。
  报告 Brier、Log loss、校准、共同样本数量及按系列赛等权的损失。

同一回合后的结构汇总以“下一回合冻结结束”为观测 checkpoint，不声称这些统计在上一回合结束瞬间就已完整可用。
旧回合结构缓存的击杀、补枪、伤害细节尚待更多原始 demo 逐项核验；新 checkpoint 对照仍是诊断，不是最终验收。

## 兼容性与旧产物

- 不覆盖旧缓存和旧实验结果；新缓存采用 v2 路径，显式重建，默认拒绝覆盖已有输出。
- `edge.py`、`edge2.py`、`mapwin_accumulate.py` 的旧加载链已明确阻断：它们从稀疏的 30 秒记录反推最终比分，
  且会串接同系列赛不同地图。须先从完整单图回合历史重建标签再接新状态，不能因上游修好就继续运行旧评估。
- 暂不改变 Map 1 赛前工作台模型，也不自动启动任何采集。
- 整场结束加延迟仍是历史结果可用时间代理，不是真实游戏/盘口接收时间；所有模型成绩不等于可执行净盈利。

## 运行

先跑测试：

```powershell
.venv\Scripts\python.exe -B -m unittest discover -s tests -q
```

30 秒缓存必须显式重建，旧缓存不会自动迁移。全量重建会解析本地 demo，可能耗时较长：

```powershell
.venv\Scripts\python.exe -B -m cs2ml.midround --rebuild
.venv\Scripts\python.exe -B -m cs2ml.midround --evaluate
```

事件级新缓存同样需要显式解析，默认写入 `data/inround_v2/`，已有输出默认拒绝覆盖：

```powershell
.venv\Scripts\python.exe -B -m cs2ml.inround
.venv\Scripts\python.exe -B -m cs2ml.inround_model --data-dir data/inround_v2 --metadata-path data/map1/history.parquet
```

缺少 v2 文件不会自动回退旧数据。仅供旧数据问题诊断的 `--allow-legacy-timing` 必须显式指定，
它不能修复旧同 tick 语义，也不能将旧状态标记为已通过时间验证。

修正后的地图 checkpoint 实验只读现有历史/回合缓存。`--output` 必须是新路径：

```powershell
.venv\Scripts\python.exe -B -m cs2ml.live_map_model --output data/map1/research/new-inplay-run/map-checkpoints.json
```

本轮已跑通的 checkpoint 诊断保存在 `data/map1/research/inplay-fixes-20260916/map-checkpoints.json`。
它使用已有历史，不是新增未见时间区间。没有对照同期可成交盘口，不进行盈利推断。

真实 demo 抽查发现冻结/回合结束编号并非总能可靠对应。例如 Falcons–G2 Map 1 inferno 存在
未确认回合结束记录和下半场边界错位；30 秒版本仅保留可确认的窗口，事件级版本严格拒绝该图。
不能为提高覆盖率猜测重新编号。该检查也意味着旧缓存的“比分内部自洽”不等于所有原始事件均已验证正确。

全量 v2 缓存和最终新时间区间验收仍待开展；本轮实际 demo 检查与单元测试不替代这些步骤。

其他实际 demo 冒烟：Astralis–FURIA Overpass 提取 21 个可信 30 秒状态；MOUZ–NRG Dust2
提取 172 个 v2 事件状态、22 个回合，全部通过非终局与快照完整性检查。均为只读提取，未写入旧/新全量缓存。

实时来源问题与使用许可见 [赛中验证契约](live-validation-contract.md)。
