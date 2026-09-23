import unittest

from cs2ml.trend_protocol import (PROTOCOL_ID, net_swing_pnl, protocol_hash,
                                  validate_session_spec)


def spec():
    return {"schema_version": 1, "protocol_id": PROTOCOL_ID,
            "session_id": "match-1-map-1", "match_id": "csgo_mc_1",
            "event_id": "123", "teams": ["A", "B"],
            "maps": [{"bout_num": 1, "map_name": "de_mirage",
                      "market_id": "9", "market_name": "Map 1 Winner"}],
            "sources": {"fivee_events": ["events.jsonl"],
                        "fivee_states": ["states.jsonl"],
                        "market": {"format": "raw_depth", "events": "market.jsonl",
                                   "metadata": "meta.json", "summary": "summary.json"}}}


class TrendProtocolTests(unittest.TestCase):
    def test_protocol_is_stable_and_spec_is_strict(self):
        self.assertEqual(len(protocol_hash()), 64)
        value = spec()
        self.assertIs(validate_session_spec(value), value)

    def test_ambiguous_identity_and_market_are_rejected(self):
        value = spec(); value["teams"] = ["A", "A"]
        with self.assertRaises(ValueError): validate_session_spec(value)
        value = spec(); value["maps"][0]["map_name"] = "Mirage"
        with self.assertRaises(ValueError): validate_session_spec(value)
        value = spec(); value["sources"]["market"]["format"] = "midpoint"
        with self.assertRaises(ValueError): validate_session_spec(value)

    def test_net_pnl_uses_explicit_cashflows(self):
        self.assertAlmostEqual(net_swing_pnl(.40, .55, .012, .013), .125)
        with self.assertRaises(ValueError): net_swing_pnl(.4, float("nan"), 0, 0)


if __name__ == "__main__":
    unittest.main()
