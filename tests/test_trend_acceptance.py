"""Phase-7 acceptance runner tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cs2ml import trend_acceptance as ta
from cs2ml import trend_dataset as td
from cs2ml import trend_scheme_freeze as sf
from tests.test_trend_dataset import _write_session, _spec


def _session_dir(root: Path, name: str) -> Path:
    session = _write_session(root / name)
    spec = _spec()
    spec["session_id"] = name
    (session / "session_spec.json").write_text(
        json.dumps(spec), encoding="utf-8")
    return session


def _freeze(root: Path, sessions: list[Path], source="game_heuristic",
            outcome="Yes") -> Path:
    frames, team1s = [], []
    for s in sessions:
        spec = json.loads((s / "session_spec.json").read_text(encoding="utf-8"))
        frame, _ = td.build_rows(s, spec, root=s)
        frames.append(frame)
        team1s.append(frame["outcome"] == outcome)
    df = pd.concat(frames, ignore_index=True)
    team1 = pd.concat(team1s)
    scheme = sf.CandidateScheme(name="frozen_test", fair_source=source,
                                horizon_seconds=60)
    result = sf.run_candidate(df, scheme, team1)
    nets = result["ledger"]["net"].to_numpy() if len(result["ledger"]) else []
    selection = {"criterion": "test", "min_trades_floor": 1, "candidates": [],
                 "winner": {"name": scheme.name, "scheme_hash": scheme.scheme_hash(),
                            "scheme": {**sf.CandidateScheme(
                                name="frozen_test", fair_source=source).__dict__}},
                 "no_eligible_reason": None}
    # mechanisms tuple 需要可 JSON 化
    selection["winner"]["scheme"]["mechanisms"] = list(
        selection["winner"]["scheme"]["mechanisms"])
    path = root / "frozen_scheme.json"
    sf.freeze_scheme(selection, path)
    return path


class ProcessSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.train = [_session_dir(self.root, f"train{i}") for i in range(2)]
        self.frozen = _freeze(self.root, self.train)
        self.test_session = _session_dir(self.root, "accept0")

    def tearDown(self):
        self.tmp.cleanup()

    def test_entry_fields(self):
        frozen = sf.load_frozen(self.frozen)
        entry = ta.process_session(self.test_session, frozen,
                                   outcome_is_team1=lambda o: o == "Yes")
        self.assertEqual(entry["session_id"], "accept0")
        self.assertEqual(entry["frozen_sha256"], frozen["sha256"])
        self.assertGreater(entry["n_triggers"], 0)
        self.assertGreaterEqual(entry["n_map_units"], 1)
        self.assertEqual(len(entry["nets"]), entry["n_trades"])
        self.assertTrue(entry["inputs_sha256"]["market_events"])

    def test_wf_scheme_requires_train_rows(self):
        frozen_path = _freeze(self.root, self.train, source="wf_ridge_exit")
        frozen = sf.load_frozen(frozen_path)
        with self.assertRaisesRegex(ValueError, "wf_scheme_requires_train_rows"):
            ta.process_session(self.test_session, frozen)
        train_frames = []
        for s in self.train:
            spec = json.loads((s / "session_spec.json").read_text(encoding="utf-8"))
            f, _ = td.build_rows(s, spec, root=s)
            train_frames.append(f)
        entry = ta.process_session(
            self.test_session, frozen,
            train_rows=pd.concat(train_frames, ignore_index=True))
        self.assertGreaterEqual(entry["n_trades"], 0)  # 不崩,行数合法


class LedgerAndVerdictTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.train = [_session_dir(self.root, f"t{i}") for i in range(2)]
        # wf 方案不需要 outcome 映射,适合 runner 层测试
        self.frozen = _freeze(self.root, self.train, source="wf_ridge_exit")
        self.train_parquets = []
        for i, s in enumerate(self.train):
            spec = json.loads((s / "session_spec.json").read_text(encoding="utf-8"))
            frame, _ = td.build_rows(s, spec, root=s)
            p = self.root / f"train{i}.parquet"
            frame.to_parquet(p, index=False)
            self.train_parquets.append(p)
        self.ledger = self.root / "ledger.jsonl"
        self.crit = ta.AcceptanceCriteria(min_maps=1, min_triggers=1)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, sessions, **kwargs):
        kwargs.setdefault("train_rows_paths", self.train_parquets)
        return ta.run_acceptance(self.frozen, self.ledger, sessions,
                                 criteria=self.crit, **kwargs)

    def test_append_and_idempotent(self):
        s1 = _session_dir(self.root, "a1")
        r1 = self._run([s1])
        self.assertEqual(r1["added_sessions"], ["a1"])
        r2 = self._run([s1])
        self.assertEqual(r2["added_sessions"], [])
        self.assertEqual(r2["skipped_duplicates"], ["a1"])
        self.assertEqual(r2["verdict"]["n_sessions"], 1)  # 没翻倍

    def test_verdict_progression_and_ci(self):
        s1 = _session_dir(self.root, "v1")
        s2 = _session_dir(self.root, "v2")
        result = self._run([s1, s2])
        verdict = result["verdict"]
        self.assertIn(verdict["status"], {"in_progress", "passed", "failed"})
        if verdict["n_trades"] >= 2:
            self.assertIsNotNone(verdict["ci95_lower_bound"])

    def test_verdict_failed_when_ci_not_positive(self):
        # 手工造一个 net 全负的账本,阈值满足 -> failed
        entries = [{"session_id": "x", "n_map_units": 5, "n_triggers": 120,
                    "nets": [-0.05, -0.04, -0.06, -0.05]},
                   {"session_id": "y", "n_map_units": 5, "n_triggers": 10,
                    "nets": [-0.02, -0.03]}]
        verdict = ta.evaluate_acceptance(entries, self.crit)
        self.assertEqual(verdict["status"], "failed")
        self.assertLess(verdict["ci95_lower_bound"], 0)
        self.assertEqual(verdict["n_map_units"], 10)
        self.assertEqual(verdict["n_triggers_evaluated"], 130)

    def test_verdict_in_progress_below_thresholds(self):
        entries = [{"session_id": "x", "n_map_units": 1, "n_triggers": 5,
                    "nets": [0.1, 0.2]}]
        crit = ta.AcceptanceCriteria(min_maps=5, min_triggers=100)
        verdict = ta.evaluate_acceptance(entries, crit)
        self.assertEqual(verdict["status"], "in_progress")

    def test_frozen_change_requires_new_round(self):
        s1 = _session_dir(self.root, "c1")
        self._run([s1])
        other = _freeze(self.root, self.train, source="wf_xgboost_exit")
        s2 = _session_dir(self.root, "c2")
        with self.assertRaisesRegex(ValueError, "frozen_scheme_changed"):
            ta.run_acceptance(other, self.ledger, [s2], criteria=self.crit,
                              train_rows_paths=self.train_parquets)
        result = ta.run_acceptance(other, self.ledger, [s2],
                                   criteria=self.crit, new_round=True,
                                   train_rows_paths=self.train_parquets)
        self.assertTrue(result["verdict"]["n_sessions"] >= 1)
        archives = list(self.root.glob("ledger.jsonl.archived-*"))
        self.assertEqual(len(archives), 1)  # 旧账本被归档而非删除
        self.assertTrue(self.ledger.is_file())


if __name__ == "__main__":
    unittest.main()
