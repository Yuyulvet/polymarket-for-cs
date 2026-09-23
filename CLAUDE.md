# CS2 Polymarket 交易研究

## 当前工作方式（2026-09-23 起）：人机分工选品
- 全自动策略方向已系统性证伪（二十多个负结果，见 memory）；**不要主动建议回退**。
- 交易决策工作流：用户选品（只大赛、大差距/高把握）→ `python -m cs2ml.match_desk --team-a X --team-b Y --map Z [--price-a .. --price-b ..]` 出读局报告 → **用户自行决定买卖** → `--record --pick a|b|none --note "理由"` 记进 `reports/match_desk/ledger.csv`。
- 用户解释决策理由时，把提炼出的规则追加到 memory 的 `cs2-desk-playbook.md`（带日期）；赛后用户回填 result_a/settled_pnl，之后可按价格带/偏差方向/笔记关键词分组核命中率 vs 60% 盈亏线（修正腿 +4.4/−6.6pt）。
- 用户的情感定价假设：市场价含粉丝情感资金，|模型p − 价格|≥5pt 是观测窗。

## 工程纪律
- venv 在 `.venv/`；一律 `python -m cs2ml.<module>` 运行；中文交流。
- demo↔市场 join 的两个对齐 bug 已记录在 memory（文件名顺序≠roster、start_date 是排期时间）——任何 join 先查 `cs2-demo-market-join-bugs`。
- WAF 不解（HLTV 403）；数据缓存优先复用 `data/map1/history.parquet`、`data/map1/features.parquet`、`reports/pistol_model_v3/samples.csv`。
