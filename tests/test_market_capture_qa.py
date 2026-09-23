from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cs2ml.market_capture_qa import analyze_capture


def row(ts, bid, ask, *, depth=True, side=None):
    value = {"type": "book" if side is None else "trade", "local_ts": ts,
             "source_ts": ts - .01, "token": "t", "best_bid": bid,
             "best_ask": ask, "trade_side": side}
    if depth:
        value.update(bid_size_l1=10, ask_size_l1=10,
                     bid_depth_l5=50, ask_depth_l5=50)
    return value


def capture(rows):
    temp = tempfile.TemporaryDirectory()
    root = Path(temp.name)
    (root / "events.jsonl").write_text(
        "\n".join(json.dumps(value) for value in rows) + "\n", encoding="utf-8")
    return temp, root


class CaptureQATests(unittest.TestCase):
    def test_pass_capture_metrics(self):
        temp, root = capture([row(i, .40 + i * .001, .42 + i * .001)
                              for i in range(20)])
        try:
            report = analyze_capture(root)
        finally:
            temp.cleanup()
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["metrics"]["local_receive_timestamp_monotonic"])
        self.assertEqual(report["metrics"]["valid_l5_depth_pct"], 1)

    def test_missing_depth_warns_without_silent_exclusion(self):
        temp, root = capture([row(i, .4 + i * .001, .42 + i * .001, depth=False)
                              for i in range(10)])
        try:
            report = analyze_capture(root)
        finally:
            temp.cleanup()
        self.assertEqual(report["status"], "WARN")
        self.assertIn("valid_l5_depth_below_warn_threshold",
                      report["exclusion_reasons"])
        self.assertEqual(report["analysis_eligibility"], "sensitivity_analysis_only")

    def test_nonmonotonic_timestamp_fails(self):
        temp, root = capture([row(2, .4, .42), row(1, .41, .43)])
        try:
            report = analyze_capture(root)
        finally:
            temp.cleanup()
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("local_receive_timestamp_not_monotonic",
                      report["exclusion_reasons"])


if __name__ == "__main__":
    unittest.main()
