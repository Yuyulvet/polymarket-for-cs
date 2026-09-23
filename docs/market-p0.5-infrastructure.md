# Market research infrastructure P0.5

P0.5 builds collection and research infrastructure. It does not train a market
model, optimize thresholds, promote a strategy, or place an order.

## Automated collection

`python -m cs2ml.market_collector --discover-only` lists current and near-term
CS2 markets. Without `--discover-only`, each market is claimed by a deterministic
identity derived from event ID, market ID and all token IDs. An existing capture
directory is a hard duplicate-prevention boundary.

The lifecycle is `DISCOVERED → SCHEDULED → RECORDING → CLOSED → FINALIZED`, with
`RECONNECTING`, `FAILED` and `INCOMPLETE` recorded as explicit states. Every
transition is appended to `audit.jsonl`. Immutable discovery metadata is written
once; actual capture start/end, counters, hashes and terminal state are written
to `manifest.json`. Closure is determined from Polymarket event/market metadata,
not a public CS2 stream.

## Capture QA

Every capture containing an events file receives `qa_report.json`. Thresholds
are frozen in `cs2ml/market_capture_qa_v1.json` and copied into the report.

- PASS: main analysis
- WARN: sensitivity analysis only
- FAIL: retained but excluded with reasons

The initial thresholds are operational integrity gates, not alpha parameters.
Changing them requires a new config version.

## Frozen alpha protocol

`docs/market-alpha-protocol-v1.md` freezes the primary 1-second / 2-cent /
5-second hypothesis. All other windows, thresholds, horizons, latency points,
taxonomy slices and CS2-alignment slices are exploratory. Discovery results do
not modify the primary hypothesis.

`python -m cs2ml.alpha_decay <capture.jsonl>` evaluates executable entry at the
first quote received at or after `decision + latency`. A missing quote is no
fill; earlier quotes and nearest-future backfills are prohibited.

## Taxonomy and CS2 annotation

`shock_taxonomy.py` produces deterministic real-time descriptive classes.
`TYPE_4_TRANSIENT_OR_REVERSAL` is an explicit ex-post label and never a feature.
`TYPE_1` is named an aggressive-flow *candidate*; it does not claim informed
trading is known.

`cs2_shock_annotation.py` attaches only identity-matched predictions whose
information timestamp is at or before the shock decision. Missing bindings and
future-only predictions produce null fields plus a reason. Annotations are
descriptive and do not trigger trades.

## Unified dataset

`python -m cs2ml.market_research_dataset <capture.jsonl> --qa-report <report>`
creates a timestamped dataset. One row is one market-level shock; complementary
tokens do not inflate sample size. The schema separates:

- `FEATURES_AVAILABLE_AT_DECISION`
- `EXECUTION_INFORMATION`
- `FUTURE_LABELS`
- non-model research metadata

Only the first group may be supplied to future training code. The output records
its DISCOVERY/VALIDATION/FORWARD_PAPER partition, protocol hash, code hash and
input hashes. Random train/test splitting is outside this contract.

## Current diagnostic status

The available schema-v2 single-market legacy capture is WARN: two-sided quotes
are present, but L1/L5 depth is absent and duplicate-content rate exceeds the v1
warning threshold. It is sensitivity-only. There are no PASS P0.5 captures yet,
so the frozen primary hypothesis remains not tested on the main-analysis set.
