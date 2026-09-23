from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import pandas as pd

from cs2ml.inplay_map_research import CATEGORICAL, VARIANTS, fit_variant_bundle
from cs2ml.live_predict import predict_row, run, state_to_frame


def training():
    rows = []
    for index in range(8):
        gap = -3 + index
        rows.append({"match_id": f"s{index//2}", "map_id": f"m{index}",
                     "round_num": 10., "map_name": "de_dust2",
                     "map_side": "de_dust2|A_CT", "score_gap": float(gap),
                     "score_gap_x_round": float(gap * 10), "a_is_ct": 1.,
                     "alive_gap": float(gap % 3 - 1), "hp_gap": float(gap * 20),
                     "y": int(gap > 0)})
    return pd.DataFrame(rows)


def paired_row():
    return {
        "decision_eligible": True, "state_timing_quality": "mqtt_push_receive_time",
        "state_source_lag_seconds": 1.2, "event_id": "e", "market_id": "m",
        "condition_id": "c", "market": "Map 1 Winner", "match_id": "match",
        "bout_num": 1, "state_hash": "hash", "state_recv_utc": "2026-01-01T00:00:00Z",
        "effective_decision_ts": 100., "focal_team": "Alpha", "focal_outcome": "Alpha",
        "focal_token": "token", "opponent_team": "Beta",
        "features": {"map_name": "Dust2", "round_number": 10,
                     "focal_is_ct": True, "focal_score": 5, "opponent_score": 4,
                     "score_diff": 1, "focal_alive": 4, "opponent_alive": 3,
                     "alive_diff": 1, "hp_diff": 80},
        "quote_at_effective_decision": {"bid": .52, "ask": .54, "mid": .53},
    }


class LivePredictTests(unittest.TestCase):
    def setUp(self):
        data = training()
        self.artifact = {"paper_only": True, "promoted": False,
                         "research_version": "test", "trained_before": "2026-01-01",
                         "variants": {name: fit_variant_bundle(
                             data, VARIANTS[name], CATEGORICAL[name])
                             for name in ("fivee_score", "fivee_state")}}

    def test_live_row_maps_to_historical_contract(self):
        frame = state_to_frame(paired_row())
        self.assertEqual(frame.iloc[0].map_name, "de_dust2")
        self.assertEqual(frame.iloc[0].map_side, "de_dust2|A_CT")
        self.assertEqual(frame.iloc[0].score_gap_x_round, 10)

    def test_stale_mismatched_or_terminal_rows_fail_closed(self):
        for mutate, reason in (
            (("state_source_lag_seconds", 8), "source_lag"),
            (("features.round_number", 11), "round_number_score"),
            (("features.opponent_alive", 0), "terminal_or_unverifiable"),
        ):
            row = paired_row()
            key, value = mutate
            if key.startswith("features."):
                row["features"][key.split(".", 1)[1]] = value
            else:
                row[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, reason):
                state_to_frame(row)

    def test_prediction_is_paper_only_and_run_writes_audit(self):
        prediction = predict_row(paired_row(), self.artifact)
        self.assertTrue(0 <= prediction["model_probability_state"] <= 1)
        self.assertIsNone(prediction["trade_action"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paired = root / "paired.jsonl"
            paired.write_text(json.dumps(paired_row()) + "\n", encoding="utf-8")
            models = root / "models.joblib"
            joblib.dump(self.artifact, models)
            output = root / "output"
            report = run(paired_path=paired, models_path=models, output_dir=output)
            self.assertEqual(report["predictions"], 1)
            self.assertFalse(report["profitability_evaluated"])
            self.assertTrue((output / "predictions.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
