from __future__ import annotations

import copy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cs2ml.map1_data import (FEATURES, build_features, features_at, first_ct_is_ct,
                            load_history, roster_key, stats_names, summarize_map, terminal_score, utc)
from cs2ml.map1_market import bind_market, proposal, team_key, validate_context, walk_book
from cs2ml.map1_model import forward_comparison, training_rows, walk_forward
from cs2ml.map1_paper import replay
from cs2ml.map1 import capture_once

A = [str(76561198000000000 + i) for i in range(1, 6)]
B = [str(76561198000000000 + i) for i in range(6, 11)]
RA, RB = roster_key(A), roster_key(B)
T = pd.Timestamp("2026-09-01T10:00:00Z")
CONDITION = "0x" + "a" * 64


def rounds(winners, mid="m1", map_no=1):
    return pd.DataFrame([{
        "demo_path": f"/{mid}/a-vs-b-m{map_no}-mirage.dem", "match_id": mid,
        "map_name": "de_mirage", "round_num": i,
        "ct_roster": RA if i <= 12 else RB,  # intentionally emulate legacy OT cache
        "t_roster": RB if i <= 12 else RA,
        "winner_side": "CT" if (winner == "a") == first_ct_is_ct(i) else "T",
    } for i, winner in enumerate(winners, 1)])


def history_row(mid="m1", start=T, winner="a"):
    row = summarize_map(rounds([winner] * 13, mid))
    row.update(start_at=start, available_at=start + pd.Timedelta(hours=3))
    for side in ("a", "b"):
        for stat in stats_names():
            row[f"{stat}_{side}"] = 20. if stat != "rounds_played" else 65.
    return row


def context():
    return {"event_id": "123", "map_no": 1, "map_name": "de_mirage",
            "map_known_at": (T - pd.Timedelta(minutes=10)).isoformat(), "map_source": "fixture:veto",
            "roster_known_at": (T - pd.Timedelta(minutes=10)).isoformat(), "roster_source": "fixture:lineup",
            "scheduled_start_at": (T + pd.Timedelta(hours=1)).isoformat(),
            "team_a": {"outcome": "Alpha Academy", "roster": A},
            "team_b": {"outcome": "Beta", "roster": B}}


def event():
    return {"id": "123", "closed": False, "markets": [{
        "id": "321", "sportsMarketType": "child_moneyline", "question": "Counter-Strike: Alpha Academy vs Beta - Map 1 Winner",
        "groupItemTitle": "Map 1 Winner", "active": True, "closed": False,
        "acceptingOrders": True, "enableOrderBook": True, "negRisk": False,
        "gameStartTime": context()["scheduled_start_at"], "conditionId": CONDITION,
        "outcomes": json.dumps(["Beta", "Alpha Academy"]), "clobTokenIds": '["22", "11"]',
        "description": "Map 1 winner; incomplete resolves 50-50.", "secondsDelay": 1,
        "feesEnabled": True, "feeSchedule": {"rate": .05, "exponent": 1, "takerOnly": True},
    }]}


def book(token, at=T, bid=.39, ask=.4, size=100):
    return {"asset_id": token, "market": CONDITION, "timestamp": str(int(at.timestamp() * 1000)),
            "min_order_size": "5", "tick_size": ".01",
            "bids": [{"price": str(bid), "size": str(size)}],
            "asks": [{"price": str(ask), "size": str(size)}]}


def snapshot(at=T, p=.7):
    return {"event_id": "123", "received_at": at.isoformat(), "status": "ready",
            "model_target": "map1_winner", "model_fitted_at": at.isoformat(),
            "train_max_available_at": (T - pd.Timedelta(days=1)).isoformat(),
            "context": context(), "p_model": p, "binding": bind_market(event(), context(), at),
            "books": {"a": book("11", at), "b": book("22", at, .59, .6)}}


def resolution(a=1., at=T + pd.Timedelta(hours=4)):
    return {"event_id": "123", "condition_id": CONDITION, "resolved_at": at.isoformat(),
            "payouts": {"11": a, "22": 1 - a}, "source": "fixture:official-resolution"}


class DataTests(unittest.TestCase):
    def test_loader_keeps_maps_separate_and_uses_end_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            m1, m2 = rounds(["a"]*13, "hltv-123-1"), rounds(["b"]*13, "hltv-123-1", 2)
            pd.concat([m1, m2]).to_parquet(root / "round_dataset.parquet")
            players = []
            for frame in (m1, m2):
                for roster in (RA, RB):
                    players.append({"demo_path": frame.iloc[0].demo_path, "roster_key": roster,
                                    **{k: 20. for k in stats_names()}})
            pd.DataFrame(players).to_parquet(root / "player_features.parquet")
            with closing(sqlite3.connect(root / "cs2.db")) as con:
                con.execute("create table demos (hltv_match_id integer, bo3gg_match_id integer, ambiguous integer)")
                con.execute("create table matches (id integer, start_date text, end_date text)")
                con.execute("insert into demos values (123, 1, 0)")
                con.execute("insert into matches values (1, ?, ?)", (T.isoformat(), (T+pd.Timedelta(hours=3)).isoformat()))
                con.commit()
            data, audit = load_history(root)
            self.assertEqual(data.map_no.tolist(), [1, 2])
            self.assertEqual(data.y.tolist(), [1, 0])
            self.assertTrue(data.available_at.eq(T+pd.Timedelta(hours=3, minutes=5)).all())
            self.assertEqual(audit["usable_map1"], 1)

    def test_team_m80_not_parsed_as_map_number(self):
        d = rounds(["a"] * 13)
        d["demo_path"] = "/series/g2-vs-m80-m1-mirage.dem"
        self.assertEqual(summarize_map(d)["map_no"], 1)

    def test_regulation_labels_and_roster_orientation(self):
        r = summarize_map(rounds(["a"] * 13))
        self.assertEqual((r["y"], r["round_wins_a"], r["round_wins_b"]), (1, 13, 0))
        self.assertFalse(r["overtime"])

    def test_overtime_does_not_end_at_13(self):
        winners = ["a", "b"] * 12 + ["a", "b", "a", "a", "a"]
        r = summarize_map(rounds(winners))
        self.assertEqual((r["round_wins_a"], r["round_wins_b"]), (16, 13))
        self.assertTrue(r["overtime"])
        self.assertFalse(terminal_score(13, 12))
        self.assertFalse(terminal_score(15, 15))

    def test_multiple_overtime_and_side_carry(self):
        self.assertEqual([first_ct_is_ct(i) for i in (12, 13, 25, 28, 31, 34, 37)],
                         [True, False, False, True, True, False, False])
        r = summarize_map(rounds(["a", "b"] * 15 + ["b"] * 4))
        self.assertEqual((r["round_wins_a"], r["round_wins_b"], r["y"]), (15, 19, 0))

    def test_truncation_duplicates_and_mixed_maps_rejected(self):
        with self.assertRaises(ValueError):
            summarize_map(rounds(["a"] * 12))
        with self.assertRaises(ValueError):
            summarize_map(pd.concat([rounds(["a"] * 13), rounds(["b"] * 13, map_no=2)]))
        with self.assertRaises(ValueError):
            summarize_map(rounds(["a"] * 14))

    def test_future_and_same_series_do_not_change_features(self):
        h = pd.DataFrame([history_row("past", T - pd.Timedelta(days=4))])
        x, audit = features_at(h, RA, RB, "mirage", T)
        future = pd.DataFrame([history_row("future", T + pd.Timedelta(days=1))])
        x2, _ = features_at(pd.concat([h, future]), RA, RB, "mirage", T)
        self.assertEqual(x, x2)
        self.assertLess(utc(audit["max_history_available_at"]), T)
        y, _ = features_at(h, RB, RA, "mirage", T)
        np.testing.assert_allclose([x[k] for k in FEATURES], [-y[k] for k in FEATURES])
        _, audit2 = features_at(h, RA, RB, "mirage", T, "past")
        self.assertEqual(audit2["history_all_a"], 0)

    def test_no_history_finite_neutral(self):
        h = pd.DataFrame([history_row()])
        x, _ = features_at(h, RA, RB, "mirage", T)
        self.assertTrue(all(v == 0 for v in x.values()))

    def test_training_waits_for_end_and_excludes_ties(self):
        rows = [{"match_id": k, "decision_at": T - pd.Timedelta(hours=2), "available_at": at}
                for k, at in (("past", T-pd.Timedelta(seconds=1)), ("equal", T), ("ongoing", T+pd.Timedelta(hours=1)))]
        self.assertEqual(training_rows(pd.DataFrame(rows), T).match_id.tolist(), ["past"])

    def test_walk_forward_frozen_holdout(self):
        h = pd.DataFrame([history_row(f"m{i}", T+pd.Timedelta(days=i), "a" if i % 2 else "b") for i in range(15)])
        f = build_features(h)
        out = walk_forward(f, min_train=4)
        valid = out.dropna(subset=["p_model"])
        self.assertTrue((pd.to_datetime(valid.train_max_available_at) < valid.decision_at).all())
        holdout = valid[valid.partition.eq("holdout")]
        self.assertEqual(holdout.n_train.nunique(), 1)
        self.assertTrue(len(holdout) > 1)


class MarketTests(unittest.TestCase):
    def test_manual_fallback_cannot_treat_tba_as_a_map(self):
        for placeholder in ("TBA", "de_tba", "TBD", "default", "To Be Announced"):
            with self.subTest(placeholder=placeholder):
                c = context(); c["map_name"] = placeholder
                with self.assertRaisesRegex(ValueError, "confirmed map"):
                    validate_context(c, T)

    def test_token_direction_and_academy_identity(self):
        b = bind_market(event(), context(), T)
        self.assertEqual((b["token_a"], b["token_b"]), ("11", "22"))
        self.assertNotEqual(team_key("Alpha"), team_key("Alpha Academy"))
        c = context(); c["team_a"]["outcome"] = "Alpha"
        with self.assertRaises(ValueError):
            bind_market(event(), c, T)

    def test_no_match_winner_fallback_or_ambiguous_map(self):
        e = event(); e["markets"][0]["sportsMarketType"] = "moneyline"
        with self.assertRaises(ValueError):
            bind_market(e, context(), T)
        e = event(); e["markets"].append(copy.deepcopy(e["markets"][0]))
        with self.assertRaises(ValueError):
            bind_market(e, context(), T)

    def test_future_veto_schedule_and_missing_fee_block(self):
        c = context(); c["map_known_at"] = (T+pd.Timedelta(seconds=1)).isoformat()
        with self.assertRaises(ValueError):
            validate_context(c, T)
        e = event(); e["markets"][0]["gameStartTime"] = (T+pd.Timedelta(hours=2)).isoformat()
        with self.assertRaises(ValueError):
            bind_market(e, context(), T)
        e = event(); e["markets"][0].pop("feeSchedule")
        with self.assertRaises(ValueError):
            bind_market(e, context(), T)

    def test_depth_weighted_fee_and_limit(self):
        b = book("11"); b["asks"] = [{"price": ".5", "size": "5"}, {"price": ".4", "size": "5"}]
        fill = walk_book(b, 10, .05)
        self.assertAlmostEqual(fill["vwap"], .45)
        self.assertAlmostEqual(fill["fee"], 5*.05*.4*.6 + 5*.05*.5*.5)
        with self.assertRaises(ValueError):
            walk_book(b, 10, .05, limit=.4)

    def test_independent_opposite_book_and_staleness(self):
        s = snapshot(p=.2)
        self.assertEqual(proposal(s)["token"], "22")
        s["books"]["b"] = book("22", T-pd.Timedelta(seconds=11), .59, .6)
        self.assertEqual(proposal(s)["action"], "skip")

    def test_wrong_model_no_cash_depth_and_crossed_book(self):
        s = snapshot(); s["model_target"] = "match_winner"
        self.assertEqual(proposal(s)["reason"], "model_target_mismatch")
        s = snapshot(); s["books"]["a"]["bids"][0]["price"] = ".5"
        self.assertEqual(proposal(s)["reason"], "crossed_book")
        s = snapshot(); s["books"]["a"]["asks"][0]["size"] = "1"
        self.assertEqual(proposal(s)["action"], "skip")
        s = snapshot(); s["event_id"] = "999"
        self.assertEqual(proposal(s)["reason"], "snapshot_event_mismatch")


class PaperTests(unittest.TestCase):
    def test_capture_blocks_unknown_map_before_network(self):
        c = context(); c["map_known_at"] = (pd.Timestamp.now(tz="UTC")+pd.Timedelta(days=1)).isoformat()
        with patch("cs2ml.map1.fetch_event") as fetch:
            result = capture_once(c, pd.DataFrame(), pd.DataFrame())
        fetch.assert_not_called()
        self.assertEqual(result["status"], "blocked")

    def test_capture_to_replay_end_to_end_without_network(self):
        h = pd.DataFrame([history_row(f"m{i}", T-pd.Timedelta(days=20-i), "a" if i%2 else "b") for i in range(10)])
        f = build_features(h)
        with patch("cs2ml.map1.now", return_value=T.isoformat()), \
             patch("cs2ml.map1.fetch_event", return_value=event()), \
             patch("cs2ml.map1.fetch_book", side_effect=lambda token: book(token, T, .39 if token=="11" else .59, .4 if token=="11" else .6)):
            s = capture_once(context(), h, f, min_train=4)
        self.assertEqual(s["status"], "ready", s.get("reason"))
        self.assertEqual(s["binding"]["token_a"], "11")
        self.assertTrue(0 < s["p_model"] < 1)
        r = replay([s], [resolution()])
        self.assertEqual(r["realized_pnl"], 0)
        self.assertEqual(r["probability_comparison_first_valid_snapshot_per_event"]["n"], 1)

    def test_delay_then_actual_fill_and_win_settlement(self):
        result = replay([snapshot(), snapshot(T+pd.Timedelta(seconds=1)), snapshot(T+pd.Timedelta(seconds=2))], [resolution()])
        self.assertEqual([r["type"] for r in result["ledger"]], ["pending", "fill", "settle"])
        self.assertAlmostEqual(result["cash"], 1000 + 10 - 4 - .12)
        self.assertEqual(result["open_cost_basis"], 0)

    def test_move_after_signal_cannot_fill_old_price(self):
        later = snapshot(T+pd.Timedelta(seconds=2)); later["books"]["a"] = book("11", T+pd.Timedelta(seconds=2), .44, .45)
        result = replay([snapshot(), later])
        self.assertEqual(result["cash"], 1000)
        self.assertEqual(result["ledger"][-1]["reason"], "insufficient_depth_at_limit")

    def test_void_not_refund_and_unresolved_not_profit(self):
        snaps = [snapshot(), snapshot(T+pd.Timedelta(seconds=2))]
        open_result = replay(snaps)
        self.assertEqual(open_result["realized_pnl"], 0)
        self.assertAlmostEqual(open_result["open_cost_basis"], 4.12)
        void = replay(snaps, [resolution(.5)])
        self.assertAlmostEqual(void["realized_pnl"], .88)
        self.assertEqual(void["probability_comparison_first_valid_snapshot_per_event"]["n"], 0)

    def test_no_fill_without_post_delay_book_or_cash(self):
        r = replay([snapshot()])
        self.assertEqual(r["pending_count"], 1)
        self.assertEqual(r["cash"], 1000)
        r = replay([snapshot(), snapshot(T+pd.Timedelta(seconds=2))], initial_cash=1)
        self.assertEqual(r["ledger"][-1]["reason"], "insufficient_paper_cash")
        r = replay([snapshot(), snapshot(T+pd.Timedelta(seconds=30))])
        self.assertEqual(r["ledger"][-1]["reason"], "missing_timely_post_delay_book")

    def test_settlement_identity_repeated_snapshots_and_loss(self):
        snaps = [snapshot(), snapshot(), snapshot(T+pd.Timedelta(seconds=2))]
        bad = resolution(); bad["condition_id"] = "wrong"
        with self.assertRaises(ValueError):
            replay(snaps, [bad])
        r = replay(snaps, [resolution(0)])
        self.assertAlmostEqual(r["realized_pnl"], -4.12)
        self.assertEqual(sum(x["type"] == "fill" for x in r["ledger"]), 1)

    def test_comparison_includes_nontraded_forecast(self):
        r = replay([snapshot(p=.4)], [resolution()])
        self.assertEqual(r["cash"], 1000)
        self.assertEqual(r["probability_comparison_first_valid_snapshot_per_event"]["n"], 1)

    def test_hybrid_does_not_train_on_unresolved_forecasts(self):
        obs = [{"event_id": str(i), "decision_at": (T+pd.Timedelta(minutes=i)).isoformat(),
                "resolved_at": (T+pd.Timedelta(days=1)).isoformat(), "payout_a": i % 2,
                "p_model": .6, "p_market": .5} for i in range(5)]
        result = forward_comparison(obs, min_train=2)
        self.assertEqual(result["hybrid_common_sample"]["hybrid"]["n"], 0)


if __name__ == "__main__":
    unittest.main()
