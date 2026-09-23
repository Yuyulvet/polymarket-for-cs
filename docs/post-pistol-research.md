# Post-pistol conversion research

Status: diagnostic only. No model is promoted and no live trading is enabled.

## External reference

The design review used the MIT-licensed
[`TaiZo1/cs2-win-prediction`](https://github.com/TaiZo1/cs2-win-prediction)
repository at commit `aba1689661219c085e2a3515d7086c5e65712325`.
No source code was copied. The adopted concepts are separating pistol,
post-pistol and normal-round regimes; observing equipment after freeze time;
and evaluating on later whole matches.

The reference project reports that pistol rounds are only weakly predictable,
while post-pistol rounds are strongly determined by the resulting economy. This
project re-tests that claim on its own audited demo corpus and stricter
result-availability boundary.

## Target and causal boundary

`cs2ml.post_pistol_model` predicts the winner of regulation rounds 2 and 14.
Every input is known at the start of that round:

- the actual starting side and pre-round score;
- mean five-player equipment value at the first post-freeze observation;
- the completed preceding pistol result, final player margin, outcome class and
  bomb-plant flag;
- roster and map history whose result `available_at` is strictly earlier than
  the target series start.

The current round winner and all current-round kills, bomb events and terminal
state are labels only. If any round identity or pre-round score is inconsistent,
the entire map is quarantined rather than partially salvaged.

## Development result

The v3 report contains 1,185 targets from 593 complete maps. A whole-series
forward evaluation scores 981 targets after the minimum training-history block.

- Constant 50%: Brier `0.2500`, AUC `0.5000`.
- Public pistol winner plus side: Brier `0.1712`, AUC `0.8026`, accuracy `79.61%`.
- Round-start equipment and side: Brier `0.1516`, AUC `0.8622`, accuracy `79.61%`.
- Equipment plus completed pistol context: Brier `0.1499`, AUC `0.8614`.
- Equipment, pistol context and prematch history: Brier `0.1505`, AUC `0.8625`.

Relative to information everyone already has after the pistol ends, equipment
improves AUC by `0.0597` and reduces Brier by `0.0196`, but does not change
threshold accuracy. All five chronological test blocks show better equipment
AUC and Brier than the pistol-winner-plus-side baseline.

The recent StarLadder-team diagnostic contains 132 scored targets: equipment
AUC `0.8691`, Brier `0.1464`, accuracy `81.82%`. It is a small inspected subset,
not a fresh acceptance population.

## Interpretation for trend trading

The model does not establish a tradeable edge by itself. The market observes
the pistol result immediately, so the valid incremental hypothesis is narrower:
the market may sometimes price all pistol wins similarly before fully accounting
for the actual freeze-time equipment distribution and force-buy risk.

The unchanged threshold accuracy means the value is probability ranking and
calibration: identifying unusually safe conversions and unusually dangerous
anti-eco/force-buy states. A profitable claim requires synchronized order books
showing that these probability differences remain absent from executable prices
after fees and delay.

## Next required work

1. Extend the demo extraction schema with cash, exact weapon classes, armor,
   helmets, kits, utility and roster-oriented saved equipment.
2. Re-run the same fixed forward protocol; do not tune on the StarLadder subset.
3. Obtain the same fields live. The current 5E event feed does not expose the
   freeze-time equipment vector; a legitimate GOTV relay or another structured
   source is required for automatic use.
4. During live paper observation, record game-event time, receipt time, feature
   completion time, market receipt time and first executable price.
5. Test whether equipment-model probability predicts subsequent executable
   Map Winner price movement beyond the pistol-result baseline.

