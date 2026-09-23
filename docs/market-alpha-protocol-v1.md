# CS2 Polymarket market-alpha protocol v1

Status: frozen discovery protocol. Any material change requires a new document
(`v2`, `v3`, …); this file and results produced under it are immutable.

## Safety and scope

- Research and paper simulation only.
- `paper_only=true`, `promoted=false`, `live_trading_enabled=false`.
- No wallet, private key, order signing, or real-money order placement.
- Local receive time is the decision and ordering clock. Source timestamps are
  supplementary evidence only.
- Missing fees and slippage remain unknown. They are never replaced with zero.

## Primary hypothesis

After a one-second absolute midpoint shock of at least 0.02, a five-second
executable residual drift may remain.

- shock window: 1 second
- shock threshold: 0.02
- holding horizon: 5 seconds from decision time
- decision: the locally received quote that establishes the shock
- entry after latency: first valid executable quote received at or after
  `decision_time + latency`
- upward side: buy at ask, exit at bid
- downward side: sell at bid and cover at ask in the paper-only symmetric
  cash-flow representation
- no quote at/after the target within the frozen quote-wait budget: unavailable
  / no fill; no nearest-future or backward fill
- quote-wait budget: 1.0 second
- primary metric: mean executable PnL before unknown fees and slippage
- required companion metrics: median, hit rate, profit factor, series/match
  clustered interval, market-level shocks, maps, matches, series, unique teams,
  and decision days

The primary hypothesis is not changed in response to discovery results.

## Exploratory analyses

The following are descriptive and explicitly `EXPLORATORY`:

- shock windows: 1, 2, 5 seconds
- thresholds: 0.01, 0.02, 0.03, 0.05
- horizons: 1, 2, 3, 5, 10, 30, 60 seconds
- latency: 0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10 seconds
- deterministic shock taxonomy and CS2-prior alignment slices

No best exploratory combination is automatically promoted or copied into the
primary hypothesis.

## Sample unit and uncertainty

One dataset row is one market-level shock. Complementary YES/NO tokens are
execution views of the same shock, not independent shocks. Tick observations
are not sample size. Bootstrap resampling is by series, then match, then event.
Fewer than five independent clusters produces no numerical interval and is
reported as `insufficient_clusters_minimum_5`.

## Capture quality policy

- PASS: eligible for main analysis.
- WARN: sensitivity analysis only.
- FAIL: excluded with explicit reasons.

Thresholds live in `cs2ml/market_capture_qa_v1.json` and are copied into every
QA report. A new threshold set requires a new config version.

## Partition contract

1. `DISCOVERY`: current and near-term collection; feature design, QA, debugging,
   descriptive analysis and hypothesis generation are allowed.
2. `VALIDATION`: begins only after protocol, features and thresholds are frozen.
   It is a strictly later time range and cannot be used to retune them.
3. `FORWARD_PAPER`: strictly later paper-bot operation using the frozen system.

There is no random train/test split. Dataset metadata must record its partition,
partition assignment time, protocol hash, input hashes, and code hashes.

## Promotion

This protocol cannot promote a strategy. Promotion requires a separate review
after adequate independent validation and forward paper evidence.
