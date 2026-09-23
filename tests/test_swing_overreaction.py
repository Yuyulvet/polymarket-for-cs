from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cs2ml.swing_overreaction import Kill, Quote, QuoteSeries, find_streaks, study


def kill(ts, team="MOUZ", eligible=True, round_num=7):
    return Kill(source_ts=ts, recv_ts=ts + 10, round_num=round_num,
                team=team, player="player", decision_eligible=eligible)


class StreakDetectionTests(unittest.TestCase):
    def test_streaks_are_maximal_and_break_on_opponent_round_or_large_gap(self):
        kills = [kill(1), kill(2), kill(3), kill(4, "NRG"), kill(20), kill(21),
                 kill(22, round_num=8)]
        chains = find_streaks(kills, min_kills=2, max_gap_seconds=5)
        self.assertEqual([[item.source_ts for item in chain] for chain in chains], [[1, 2, 3], [20, 21]])


class ExecutionStudyTests(unittest.TestCase):
    def quotes(self):
        return {
            "MOUZ": QuoteSeries([Quote(90, .49, .51), Quote(102, .59, .61), Quote(200, .40, .42)]),
            "NRG": QuoteSeries([Quote(90, .49, .51), Quote(102, .40, .42),
                                Quote(132, .47, .49), Quote(200, .58, .60)]),
        }

    def test_fade_buys_at_ask_and_sells_at_bid_without_midpoint_fill(self):
        rows = study([[kill(100), kill(101)]], self.quotes(), ("MOUZ", "NRG"),
                     timing_basis="source_update_version", entry_delay=1,
                     pre_seconds=5, reaction_threshold=.03, horizons=(30,), fee_rate=0)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row["triggered"])
        self.assertAlmostEqual(row["reaction_mid_change"], .10)
        self.assertEqual((row["entry_ask"], row["exit_bid"]), (.42, .47))
        self.assertAlmostEqual(row["gross_pnl"], .05)

    def test_strict_receive_excludes_any_chain_containing_backfill(self):
        rows = study([[kill(100, eligible=False), kill(101)]], self.quotes(), ("MOUZ", "NRG"),
                     timing_basis="strict_receive", entry_delay=1, pre_seconds=5,
                     reaction_threshold=0, horizons=(30,), fee_rate=0)
        self.assertEqual(rows, [])

    def test_fee_assumption_is_applied_on_both_sides(self):
        rows = study([[kill(100), kill(101)]], self.quotes(), ("MOUZ", "NRG"),
                     timing_basis="source_update_version", entry_delay=1,
                     pre_seconds=5, reaction_threshold=0, horizons=(30,), fee_rate=.05)
        expected = .47 - .42 - .05*.42*.58 - .05*.47*.53
        self.assertAlmostEqual(rows[0]["net_pnl"], expected)


if __name__ == "__main__":
    unittest.main()
