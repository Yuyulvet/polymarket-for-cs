# Market microstructure P0

This phase is research and paper trading only. It does not connect a wallet,
sign orders, or enable live execution. Existing experiment artifacts remain
immutable; new command-line runs create a new UTC-stamped directory and record
input and code hashes.

## Repository audit

The existing fundamental path preserves the main anti-leakage boundary:
`features.py` emits a match row before applying that match's result, and
`train.py` uses expanding time folds rather than random train/test splits. The
Map 1 and in-play paths add result-availability, stable roster/SteamID identity,
map identity, and whole-series forward-split checks. Live pairing uses local
receive time, backward-only state/quote joins, and keeps future horizons under
an `evaluation_only` boundary. These mechanisms were retained.

Two market-data defects were found:

1. `RealtimeRecorder._top()` returned `None` for every non-empty book because
   its parser was unreachable after `_source_ts()` returned.
2. The legacy raw-depth rebuilder calculated L5 depth by sorting sizes, rather
   than selecting the five best price levels.

Both are fixed with regression tests. The old `paper_trade.py` remains a legacy
midpoint baseline and is not the execution model for this work. The older trend
dataset also has a historical fee fallback to zero; Experiment 0 deliberately
does not inherit that fallback. Missing fee or slippage inputs stay unknown.

## Recorder schema v3

Normalized book records now include best bid/ask, spread, midpoint, L1 sizes,
L5 cumulative depth, and the five best levels on both sides. Trade records use
explicit `trade_price`, `trade_size`, and `trade_side` fields and carry the most
recent reconstructed book. Rows include event, market, condition, token, and
outcome identifiers where metadata is available.

`local_ts` is the research clock. `source_ts` is evidence only and never
controls ordering. The recorder derives `recv_utc` from the same local receive
timestamp so the numeric and ISO representations cannot drift across a write.

## Time-safe features

`python -m cs2ml.market_features <jsonl...>` builds causal features from
`data/realtime` records. Features include spread, midpoint, microprice, L1/L5
imbalance, 1/3/5/10/30-second trade imbalance and volume, trade intensity,
short returns, realized volatility, depth depletion, spread change,
microprice-mid divergence, velocity, and acceleration.

The implementation processes observations in local-receive order. Window
lookups are last-observation-at-or-before the boundary. Tests append extreme
future records and verify that the earlier feature prefix is byte-for-byte
unchanged after null normalization.

Legacy normalized recordings made before schema v3 lack depth sizes. Those
files can still supply spread/mid/return features, but depth, OBI, and true
liquidity strata remain unknown. They must not be presented as a complete
microstructure validation cohort.

## Experiment 0

`python -m cs2ml.market_shocks <jsonl...>` detects 1/2/5-second shocks at
1/2/3/5-cent thresholds and labels 1/2/3/5/10/30/60-second horizons.

For upward momentum, entry is the first ask at or after the latency-adjusted
decision time and exit is the first bid at or after the horizon. Downward
momentum uses the symmetric sell-at-bid / cover-at-ask paper cash flow. Reports
keep raw midpoint drift, executable PnL, spread cost, estimated fees, estimated
slippage, and net drift separate. If either fee or slippage is unknown, net
drift is unknown rather than zero-cost.

Reports show observations, distinct shocks, maps, matches, series, teams,
decision days, mean/median executable PnL, clustered 95% intervals, hit rate,
and profit factor. Bootstrap resampling uses series, then match, then event as
the fallback cluster. Direction, magnitude, starting probability, spread,
liquidity, OBI, trade imbalance, volume, favorite/underdog, and reliable market
type are retained as strata. Live stage is not inferred from an unreliable
label.

No strategy is promoted by this experiment. A confidence interval crossing
zero is reported as insufficient evidence. Even a positive interval is only an
observed research result and still requires a frozen out-of-sample forward
window.

The follow-on automated collection, QA, frozen protocol, latency, taxonomy,
annotation and unified-dataset layer is documented in
`docs/market-p0.5-infrastructure.md`.
