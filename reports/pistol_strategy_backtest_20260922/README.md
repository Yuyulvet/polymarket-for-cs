# Pistol swing strategy backtest (2026-09-22)

User strategy: model calls the pistol winner at >=55% confidence;
v1 = sell the instant the pistol ends; v3 = if the strong team loses
the pistol, hold to round-3 end when the map model still favors it
(p_mapwin >= 0.55), else exit immediately.

## Pre-registered benchmarks (do not re-litigate)
- stronger-team pistol hit-rate ceiling: 0.534
- tape legs: win +6.25pt / loss -11.2pt -> breakeven hit 0.6418
- replay EV at ceiling: -1.9pt
- R2 rebound rate after losing pistol: 0.444

A 55% trigger line has NO data support (ceiling 53.4%; v3 OOF AUC ~0.48
with >=60% subset hitting 42-43%). This run executes it anyway.

## Honesty ledger
- prices are minute-level mids: no spread, no book; cost03/fee columns
  are sensitivity only
- 'stronger team' = composite(all_win/kd/round + map_win/pistol gaps),
  90-day half-life history; HLTV crawl out of scope (user named it the
  ideal source)
- in-sample p_cal is optimistic (Platt applied to all rows); OOF is the
  honest arm. Read OOF first.
- pistol timing is anchored to the first >=3pt 1-min price jump near
  the demo-clock pistol (market-time detection); alignment_diag must
  show a high jump-side match rate before any P&L is interpreted

## Result tables
```json
{
  "universe_px_maps": 278,
  "conf": 0.55,
  "sources": {
    "insample": {
      "v3_full": {
        "universe": {
          "n_with_preds": 174,
          "n_px_overlap": 139,
          "n_triggered_strict": 73,
          "n_triggered_relaxed": 125,
          "n_side_mapping": 171,
          "skip_reasons": {
            "no_px_prices": 35
          }
        },
        "alignment_diag": {
          "clock_delta_min_median": 2.0,
          "clock_delta_min_p10": -1.2,
          "clock_delta_min_p90": 14.2,
          "jump_side_match": "107/138",
          "by_lag": {
            "lag_-2": {
              "n": 102,
              "won_pop_mean": 0.016,
              "lost_pop_mean": -0.016
            },
            "lag_-1": {
              "n": 102,
              "won_pop_mean": 0.027,
              "lost_pop_mean": -0.021
            },
            "lag_+0": {
              "n": 102,
              "won_pop_mean": 0.038,
              "lost_pop_mean": -0.049
            },
            "lag_+1": {
              "n": 102,
              "won_pop_mean": 0.045,
              "lost_pop_mean": -0.048
            },
            "lag_+2": {
              "n": 102,
              "won_pop_mean": 0.051,
              "lost_pop_mean": -0.056
            }
          },
          "tape_reference": {
            "win_pop": 6.25,
            "loss_drop": -11.2
          }
        },
        "benchmarks": {
          "ceiling_hit": 0.534,
          "breakeven_hit": 0.6418,
          "tape_win_pop_pt": 6.25,
          "tape_loss_drop_pt": -11.2,
          "replay_ev_at_ceiling_pt": -1.9,
          "r2_rebound_rate": 0.444
        },
        "arms": {
          "v1_strict": {
            "n": 65,
            "hit": 0.5385,
            "ci": [
              0.419,
              0.654
            ],
            "pnl_mean": 0.0048,
            "pnl_median": 0.02,
            "pnl_total": 0.31,
            "win_rate": 0.523,
            "pnl_cost03": {
              "n": 65,
              "pnl_mean": -0.0252,
              "pnl_median": -0.01,
              "pnl_total": -1.64,
              "win_rate": 0.446
            },
            "pnl_fee": {
              "n": 65,
              "pnl_mean": -0.0163,
              "pnl_median": -0.0047,
              "pnl_total": -1.062,
              "win_rate": 0.477
            }
          },
          "v3_strict": {
            "pnl": {
              "n": 30,
              "pnl_mean": -0.0608,
              "pnl_median": -0.0575,
              "pnl_total": -1.825,
              "win_rate": 0.2
            },
            "hold_legs": {
              "n": 8,
              "pnl_mean": -0.0919,
              "pnl_median": -0.0975,
              "pnl_total": -0.735,
              "win_rate": 0.25
            },
            "hold_n": 8,
            "rebound_hit": {
              "n": 8,
              "hit": 0.0,
              "ci": [
                0.0,
                0.324
              ],
              "note": "R2 won during hold"
            }
          },
          "v1_relaxed": {
            "n": 102,
            "hit": 0.5784,
            "ci": [
              0.481,
              0.67
            ],
            "pnl_mean": 0.0015,
            "pnl_median": 0.0,
            "pnl_total": 0.15,
            "win_rate": 0.48,
            "pnl_cost03": {
              "n": 102,
              "pnl_mean": -0.0285,
              "pnl_median": -0.03,
              "pnl_total": -2.91,
              "win_rate": 0.422
            },
            "pnl_fee": {
              "n": 102,
              "pnl_mean": -0.02,
              "pnl_median": -0.0195,
              "pnl_total": -2.037,
              "win_rate": 0.441
            }
          },
          "v3_relaxed": {
            "pnl": {
              "n": 43,
              "pnl_mean": -0.0588,
              "pnl_median": -0.06,
              "pnl_total": -2.53,
              "win_rate": 0.209
            },
            "hold_legs": {
              "n": 10,
              "pnl_mean": -0.099,
              "pnl_median": -0.1125,
              "pnl_total": -0.99,
              "win_rate": 0.2
            },
            "hold_n": 10,
            "rebound_hit": {
              "n": 10,
              "hit": 0.0,
              "ci": [
                -0.0,
                0.278
              ],
              "note": "R2 won during hold"
            }
          },
          "conf_sweep_model_hit": {
            "conf_0.50": {
              "n": 174,
              "model_hit": 0.546
            },
            "conf_0.55": {
              "n": 125,
              "model_hit": 0.52
            },
            "conf_0.60": {
              "n": 86,
              "model_hit": 0.5116
            }
          }
        }
      },
      "control_v2best": {
        "universe": {
          "n_with_preds": 174,
          "n_px_overlap": 139,
          "n_triggered_strict": 73,
          "n_triggered_relaxed": 123,
          "n_side_mapping": 171,
          "skip_reasons": {
            "no_px_prices": 35
          }
        },
        "alignment_diag": {
          "clock_delta_min_median": 2.0,
          "clock_delta_min_p10": -1.2,
          "clock_delta_min_p90": 14.2,
          "jump_side_match": "107/138",
          "by_lag": {
            "lag_-2": {
              "n": 99,
              "won_pop_mean": 0.014,
              "lost_pop_mean": -0.017
            },
            "lag_-1": {
              "n": 99,
              "won_pop_mean": 0.031,
              "lost_pop_mean": -0.019
            },
            "lag_+0": {
              "n": 99,
              "won_pop_mean": 0.042,
              "lost_pop_mean": -0.046
            },
            "lag_+1": {
              "n": 99,
              "won_pop_mean": 0.048,
              "lost_pop_mean": -0.047
            },
            "lag_+2": {
              "n": 99,
              "won_pop_mean": 0.043,
              "lost_pop_mean": -0.055
            }
          },
          "tape_reference": {
            "win_pop": 6.25,
            "loss_drop": -11.2
          }
        },
        "benchmarks": {
          "ceiling_hit": 0.534,
          "breakeven_hit": 0.6418,
          "tape_win_pop_pt": 6.25,
          "tape_loss_drop_pt": -11.2,
          "replay_ev_at_ceiling_pt": -1.9,
          "r2_rebound_rate": 0.444
        },
        "arms": {
          "v1_strict": {
            "n": 63,
            "hit": 0.4921,
            "ci": [
              0.373,
              0.612
            ],
            "pnl_mean": 0.0008,
            "pnl_median": 0.02,
            "pnl_total": 0.05,
            "win_rate": 0.524,
            "pnl_cost03": {
              "n": 63,
              "pnl_mean": -0.0292,
              "pnl_median": -0.01,
              "pnl_total": -1.84,
              "win_rate": 0.444
            },
            "pnl_fee": {
              "n": 63,
              "pnl_mean": -0.0206,
              "pnl_median": -0.0047,
              "pnl_total": -1.296,
              "win_rate": 0.476
            }
          },
          "v3_strict": {
            "pnl": {
              "n": 32,
              "pnl_mean": -0.0541,
              "pnl_median": -0.0575,
              "pnl_total": -1.73,
              "win_rate": 0.25
            },
            "hold_legs": {
              "n": 8,
              "pnl_mean": -0.0663,
              "pnl_median": -0.0675,
              "pnl_total": -0.53,
              "win_rate": 0.375
            },
            "hold_n": 8,
            "rebound_hit": {
              "n": 8,
              "hit": 0.0,
              "ci": [
                0.0,
                0.324
              ],
              "note": "R2 won during hold"
            }
          },
          "v1_relaxed": {
            "n": 99,
            "hit": 0.5455,
            "ci": [
              0.448,
              0.64
            ],
            "pnl_mean": 0.002,
            "pnl_median": 0.01,
            "pnl_total": 0.195,
            "win_rate": 0.505,
            "pnl_cost03": {
              "n": 99,
              "pnl_mean": -0.028,
              "pnl_median": -0.02,
              "pnl_total": -2.775,
              "win_rate": 0.434
            },
            "pnl_fee": {
              "n": 99,
              "pnl_mean": -0.0195,
              "pnl_median": -0.0146,
              "pnl_total": -1.926,
              "win_rate": 0.455
            }
          },
          "v3_relaxed": {
            "pnl": {
              "n": 45,
              "pnl_mean": -0.0527,
              "pnl_median": -0.06,
              "pnl_total": -2.37,
              "win_rate": 0.244
            },
            "hold_legs": {
              "n": 10,
              "pnl_mean": -0.0785,
              "pnl_median": -0.095,
              "pnl_total": -0.785,
              "win_rate": 0.3
            },
            "hold_n": 10,
            "rebound_hit": {
              "n": 10,
              "hit": 0.0,
              "ci": [
                -0.0,
                0.278
              ],
              "note": "R2 won during hold"
            }
          },
          "conf_sweep_model_hit": {
            "conf_0.50": {
              "n": 174,
              "model_hit": 0.546
            },
            "conf_0.55": {
              "n": 123,
              "model_hit": 0.5528
            },
            "conf_0.60": {
              "n": 79,
              "model_hit": 0.5949
            }
          }
        }
      }
    },
    "oof": {
      "v3_full": {
        "universe": {
          "n_with_preds": 174,
          "n_px_overlap": 139,
          "n_triggered_strict": 23,
          "n_triggered_relaxed": 43,
          "n_side_mapping": 171,
          "skip_reasons": {
            "no_px_prices": 35
          }
        },
        "alignment_diag": {
          "clock_delta_min_median": 2.0,
          "clock_delta_min_p10": -1.2,
          "clock_delta_min_p90": 14.2,
          "jump_side_match": "107/138",
          "by_lag": {
            "lag_-2": {
              "n": 36,
              "won_pop_mean": -0.004,
              "lost_pop_mean": -0.015
            },
            "lag_-1": {
              "n": 36,
              "won_pop_mean": 0.014,
              "lost_pop_mean": -0.017
            },
            "lag_+0": {
              "n": 36,
              "won_pop_mean": 0.031,
              "lost_pop_mean": -0.04
            },
            "lag_+1": {
              "n": 36,
              "won_pop_mean": 0.051,
              "lost_pop_mean": -0.045
            },
            "lag_+2": {
              "n": 36,
              "won_pop_mean": 0.057,
              "lost_pop_mean": -0.066
            }
          },
          "tape_reference": {
            "win_pop": 6.25,
            "loss_drop": -11.2
          }
        },
        "benchmarks": {
          "ceiling_hit": 0.534,
          "breakeven_hit": 0.6418,
          "tape_win_pop_pt": 6.25,
          "tape_loss_drop_pt": -11.2,
          "replay_ev_at_ceiling_pt": -1.9,
          "r2_rebound_rate": 0.444
        },
        "arms": {
          "v1_strict": {
            "n": 20,
            "hit": 0.4,
            "ci": [
              0.219,
              0.613
            ],
            "pnl_mean": -0.0035,
            "pnl_median": -0.0225,
            "pnl_total": -0.07,
            "win_rate": 0.4,
            "pnl_cost03": {
              "n": 20,
              "pnl_mean": -0.0335,
              "pnl_median": -0.0525,
              "pnl_total": -0.67,
              "win_rate": 0.4
            },
            "pnl_fee": {
              "n": 20,
              "pnl_mean": -0.0244,
              "pnl_median": -0.0462,
              "pnl_total": -0.487,
              "win_rate": 0.4
            }
          },
          "v3_strict": {
            "pnl": {
              "n": 12,
              "pnl_mean": -0.0537,
              "pnl_median": -0.035,
              "pnl_total": -0.645,
              "win_rate": 0.25
            },
            "hold_legs": {
         
```

## Findings (final, all four source/variant cells agree in sign)

**Data-integrity bugs found and fixed during this backtest (affect earlier analyses):**
1. Filename order (`a-vs-b`) does NOT reliably encode roster_a/roster_b: the
   market's resolved map winner agreed with the label-implied slug only 50%
   of the time (n=171). Any team-level market<->demo join keyed on filename
   order silently swaps ~half the teams. Fix used here: bind outcome names
   to roster sides via the settled (>=0.9) final price.
2. `demos.start_date` is the SCHEDULED series time; per-match offset to the
   real Map-1 pistol runs -4 to +45 minutes. Fix: per-match constant delta
   maximizing demo round-end (rounds 1-10) overlap with >=2pt mirror price
   jumps. After both fixes, round-1 jump direction agrees with pistol labels
   107/138 = 78%, and by-lag pops (+5.1pt won / -4.5pt lost at lag+1)
   reproduce the tape benchmark shape (+6.25 / -11.2).

**Strategy results (mid-price fills; cost03 = -3pt slip, fee = symmetric taker):**
- v1 relaxed (>=55% any direction), the most favorable arm:
  OOF hit 55.6% (n=36, CI 39.6-70.5) vs required breakeven 64.2%;
  P&L mean ~0.0pt raw, -3.1pt with 3pt slip, -2.2pt with fee.
  In-sample (optimistic) hit 57.8% (n=102), still ~0 raw, negative with costs.
- v1 strict (model pick must equal the composite stronger team):
  OOF hit 40.0% (n=20) -- the >=55% confident subset is anti-predictive,
  consistent with the pre-registered OOF AUC ~0.48.
- v3 (user's conditional rule): quick-exit legs (map read <55%) lose
  -4.7 to -7.2pt per leg (the -11.2pt loss drop, partially absorbed);
  hold legs (map read >=55%, lost pistol, hold to R3 end): 0/9 unique maps
  won R2 during the hold, mean -6 to -12pt. The "hold and wait for the
  rebound" branch did not produce a single rebound in this sample.
- Confidence sweep: model hit RATES FALL as the trigger tightens
  (56.5% @50% -> 50-53% @55% -> 50-64% noisy @60%), i.e. the model's
  confidence carries no information in OOF.

**Verdict: not profitable.** Even at the optimistic in-sample calibration and
zero trading cost, no arm clears the 64.2% breakeven hit rate; with realistic
costs every arm is negative. This confirms the pre-registered benchmarks:
the 55% trigger line sits above the measured 53.4% ceiling, and the v3
model's high-confidence subset is confident-but-wrong under honest OOF.
