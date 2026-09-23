from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from cs2ml.map1_data import roster_key
from cs2ml.pistol_model import load_event_manifest
from cs2ml.post_pistol_model import (
    ECONOMY,
    PISTOL_CONTEXT,
    extract_post_pistol_events,
    fit_model,
    model_definitions,
)


A = [str(76561198110000000 + i) for i in range(1, 6)]
B = [str(76561198220000000 + i) for i in range(1, 6)]
RA, RB = roster_key(A), roster_key(B)
T = pd.Timestamp("2026-08-10T12:00:00Z")
MANIFEST = Path(__file__).parents[1] / "cs2ml" / "starladder_fall_2026.json"


def inputs(round2_winner=RA):
    map_id = "/demos/mouz-vs-nrg-m1-mirage.dem"
    winners = [RA, round2_winner, RA, RB, RA, RB, RA, RB, RA, RB, RA, RB, RB, RB]
    rows, score_a, score_b = [], 0, 0
    for number, winner in enumerate(winners, 1):
        ct, tt = (RA, RB) if number <= 12 else (RB, RA)
        if number in (2, 14):
            ct_equip, t_equip = (4000.0, 1500.0) if number == 2 else (1500.0, 4000.0)
        else:
            ct_equip = t_equip = 1000.0
        side = "CT" if winner == ct else "T"
        rows.append({
            "demo_path": map_id, "round_num": number, "map_name": "de_mirage",
            "winner_side": side, "ct_roster": ct, "t_roster": tt,
            "roster_a": RA, "roster_b": RB, "ct_is_a": ct == RA,
            "ct_equip0": ct_equip, "t_equip0": t_equip,
            "score_a_before": score_a, "score_b_before": score_b,
            "final_margin": 4.0 if number in (1, 13) else 2.0,
            "outcome_type": "stomp" if number == 1 else (
                "comeback" if number == 13 else "even"),
            "bomb_planted": number == 1,
        })
        if winner == RA:
            score_a += 1
        else:
            score_b += 1
    history = pd.DataFrame([{
        "map_id": map_id, "match_id": "series-1", "map_name": "de_mirage",
        "roster_a": RA, "roster_b": RB, "start_at": T,
        "available_at": T + pd.Timedelta(hours=3),
    }])
    labels = pd.DataFrame([{"map_id": map_id, "complete": True,
                            "history_check": "matched"}])
    return pd.DataFrame(rows), history, labels


class PostPistolModelTests(unittest.TestCase):
    def test_extracts_round_2_and_14_with_oriented_start_state(self):
        result, audit = extract_post_pistol_events(
            *inputs(), load_event_manifest(MANIFEST))
        self.assertEqual(result.post_round.tolist(), [2, 14])
        self.assertEqual(audit["post_pistol_rows"], 2)
        self.assertEqual(result.equipment_gap.tolist(), [2500.0, 2500.0])
        self.assertEqual(result.pistol_winner_gap.tolist(), [1.0, -1.0])
        self.assertEqual(result.previous_stomp_gap.tolist(), [1.0, 0.0])
        self.assertEqual(result.previous_comeback_gap.tolist(), [0.0, -1.0])
        self.assertEqual(result.previous_bomb_t_gap.tolist(), [-1.0, 0.0])

    def test_current_round_winner_changes_label_not_features(self):
        manifest = load_event_manifest(MANIFEST)
        won, _ = extract_post_pistol_events(*inputs(RA), manifest)
        lost, _ = extract_post_pistol_events(*inputs(RB), manifest)
        won = won[won.post_round.eq(2)].iloc[0]
        lost = lost[lost.post_round.eq(2)].iloc[0]
        self.assertNotEqual(won.y, lost.y)
        for feature in [*ECONOMY, *PISTOL_CONTEXT]:
            self.assertEqual(won[feature], lost[feature], feature)

    def test_incorrect_pre_round_score_fails_closed(self):
        rounds, history, labels = inputs()
        rounds.loc[rounds.round_num.eq(14), "score_a_before"] += 1
        with self.assertRaisesRegex(ValueError, "empty_post_pistol_targets"):
            extract_post_pistol_events(rounds, history, labels,
                                       load_event_manifest(MANIFEST))

    def test_feature_sets_exclude_current_round_outcome(self):
        definitions = model_definitions()
        self.assertEqual(definitions["pistol_winner_only"], ["pistol_winner_gap"])
        for features in definitions.values():
            self.assertNotIn("y", features)
            self.assertNotIn("winner_side", features)
            self.assertNotIn("final_margin", features)

    def test_mirrored_model_is_roster_complementary(self):
        train = pd.DataFrame({
            "equipment_gap": [-3000.0, -1000.0, 1200.0, 3200.0],
            "y": [0, 0, 1, 1], "match_id": ["a", "b", "c", "d"],
            "start_at": pd.date_range("2026-01-01", periods=4, tz="UTC"),
            "target_event_team": [False] * 4,
        })
        model = fit_model(train, ["equipment_gap"],
                          pd.Timestamp("2026-02-01T00:00:00Z"))
        probabilities = model.predict_proba(np.array([[2000.0], [-2000.0]]))[:, 1]
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
