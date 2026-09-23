# 5EPlay live state integration

Validated on 2026-09-17 against the public match page for Vitality vs magic:

- Page: `https://event.5eplay.com/csgo/matches/csgo_mc_2398089`
- Initial/current snapshot: `https://esports-data.5eplaycdn.com/v1/api/csgo/matches/csgo_mc_2398089/data`
- Event log: `https://esports-data.5eplaycdn.com/v1/api/csgo/match/csgo_mc_2398089/event/log`
- Page MQTT topic: `csgo/product/detail/csgo_mc_2398089`

The page bundle renders `money`, `hp`, `weapon`, `helmet`, `kevlar`,
`has_defusekit`, `c4`, kills, deaths and assists for every player. The displayed
weapon is one icon, not a complete inventory. The feed currently does not expose
the demo parser's total `current_equip_value`, and no complete grenade inventory
was observed. Therefore the normalized schema deliberately calls it
`display_weapon`; it must not be used as an exact replacement for historical
total equipment value.

## Recorders

Use MQTT as the strict live source and run it next to the targeted Polymarket
recorder:

```powershell
.venv\Scripts\python.exe -m cs2ml.fivee_mqtt `
  --match-id csgo_mc_2398089 --hours 4

.venv\Scripts\python.exe -m cs2ml.realtime_record `
  --event 1005582 --market "Map 2 Winner" --hours 4
```

The MQTT recorder writes changed snapshots to `data/fivee/mqtt_*.jsonl` and
keeps the complete raw `match` object beside a normalized summary. Every record
has local wall-clock and monotonic receive times plus the MQTT `from_ver` and
`this_ver` markers. Its initial HTTP snapshot and first snapshot after a
reconnect are marked `decision_eligible=false`; normal MQTT pushes are eligible.
The observed `this_ver` clock is also decoded as zero-padded Unix deciseconds.
Strict pairing rejects states whose estimated source-to-receive lag exceeds five
seconds; this threshold can be changed with `--max-source-lag`.

`cs2ml.fivee_state` remains available only as a polling diagnostic. On the
Vitality vs magic observation its HTTP payload changed at an almost exact
40-second cadence and even regressed from 12-8 to 11-8. It is therefore not a
strict trading input. `cs2ml.live_pair` accepts only
`mqtt_push_receive_time` by default; `--allow-http-polling` is an explicit
diagnostic-only escape hatch.

The Polymarket event contains separate Map 1 and Map 2 winner markets. For the
supplied match, Map 2 is market `4473245`, condition
`0xa61b0987d912313dec724511fb5c1fc3471ebb73d66ac0d77447b47bb6b3d231`.
Pair on the token-specific CLOB receive timestamp, not Gamma's cached
`bestBid`/`bestAsk` values.

## Smoke observation

At `2026-09-17T14:27Z`, the recorder saw Dust2 round 12, Vitality leading 8-3,
with displayed cash totals 24,050 vs 1,100. A direct CLOB read within about one
minute showed the Vitality Map 2 token at 0.973 bid / 0.995 ask. This loose smoke
pair only verifies source identity and schema; it is not a latency measurement
or evidence of a tradable edge.

## MQTT validation

The exact detail topic was validated against live match `csgo_mc_2397677` on
2026-09-17. The broker acknowledged the subscription at QoS 0 and delivered a
`csgo-detail` payload containing round/score plus per-player cash, HP, displayed
weapon, armour and kit. A production smoke capture recorded an ineligible HTTP
seed followed by an eligible MQTT push. Reconnect snapshots remain excluded
until their timing semantics have been validated across more matches.

A ten-minute cadence capture on that match contained 15 MQTT state snapshots
(plus one HTTP seed). Inter-arrival time was 19--90 seconds, median 24 seconds.
Interpreting `this_ver` as the observed Unix-decisecond clock gave 0.9--28.9
seconds source-to-receive lag, median 9.9 seconds; only 6/15 frames were at most
five seconds old. The separately acknowledged
`csgo/product/event/log/<match_id>` subscription delivered zero messages during
the same ten-minute observation. This feed is therefore suitable for a
strictly-filtered round/state pilot, not yet for a per-kill scalping claim.

The source `curr_round_num` also disagreed with the visible score in this
capture. Normalization now derives the active round as
`team1_score + team2_score + 1` and retains the source round number and mismatch
flag for audit. Live inference additionally verifies this invariant.

Pair a capture with executable quotes using local receive time:

```powershell
.venv\Scripts\python.exe -m cs2ml.live_pair `
  --fivee data\fivee\mqtt_csgo_mc_2398089_*.jsonl `
  --quotes data\realtime\event_1005582_*.jsonl `
  --market-meta data\realtime\event_1005582_*.meta.json `
  --bout-num 2 --market "Map 2 Winner" `
  --output-dir reports\live_pair_vit_magic_map2
```

## Paper probability model

`inplay_map_research` now evaluates two variants whose inputs match the verified
5E subset: `fivee_score` (score, round and side) and `fivee_state` (adds alive
and HP gaps). On the existing historical forward diagnostic, adding alive/HP
reduced equal-series Brier loss by about 0.00563 across 200 scored series; this
is not a live-market profitability result because historical event sampling is
denser than the observed 5E push cadence.

Exported `models.joblib` files remain paper-only. Apply one only to strict paired
rows:

```powershell
.venv\Scripts\python.exe -m cs2ml.live_predict `
  --paired reports\live_pair_vit_magic_map2\paired_states.jsonl `
  --models data\map1\research\fivee-compatible-20260918-v3\models.joblib `
  --output-dir reports\live_predict_vit_magic_map2
```

The output contains model probabilities and gross model-minus-quote gaps, but
never emits a trade action. Fees, slippage, queue position and fill probability
must be added and tested on new synchronized matches first.
