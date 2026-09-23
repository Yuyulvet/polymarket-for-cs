import unittest

import numpy as np
import pandas as pd

from cs2ml.live_map_model import (STRUCT_GAPS, VARIANTS, build_snapshots,
                                  compare_variants, forward_splits, normalize_round_events)
from cs2ml.map1_data import first_ct_is_ct, summarize_map


A, B = '1,2,3,4,5', '10,6,7,8,9'


def example(winners=None, series='hltv-1', map_no=1):
    winners = winners or [A] * 12 + [B] * 12 + [A, B, A, B, A, A]
    rows = []
    for r, winner in enumerate(winners, 1):
        actual_ct = A if first_ct_is_ct(r) else B
        # Deliberately reproduce the legacy single-flip cache.
        ct, t = (A, B) if r <= 12 else (B, A)
        row = dict(demo_path=f'D:/demos/{series}/test-m{map_no}-mirage.dem',
                   match_id=series, map_name='de_mirage', round_num=r,
                   ct_roster=ct, t_roster=t, winner_side='CT' if winner == actual_ct else 'T',
                   winner_roster=winner, label_ct_win=int(winner == actual_ct), round_class='regular')
        for prefix, roster in [('ct', ct), ('t', t)]:
            row.update({f'{prefix}_equip': 5000, f'{prefix}_tier6': 'full' if r % 2 else 'eco0',
                        f'{prefix}_alive': 3, f'{prefix}_hp': 300 if roster == A else 200,
                        f'{prefix}_kills': 2, f'{prefix}_first': int(roster == winner),
                        f'{prefix}_trade': int(roster == A), f'{prefix}_util': 20,
                        f'{prefix}_eco_kills': 1})
        rows.append(row)
    d = pd.DataFrame(rows)
    h = summarize_map(d)
    h.update(start_at=pd.Timestamp('2026-01-01', tz='UTC'),
             available_at=pd.Timestamp('2026-01-01T03:00Z'))
    return d, h


def comparison_rows(n=20):
    return pd.DataFrame([dict(map_id=f'map-{i}', match_id=f'series-{i}', k=6, y=i % 2,
                              start_at=pd.Timestamp('2026-01-01', tz='UTC') + pd.Timedelta(days=i),
                              available_at=pd.Timestamp('2026-01-01T03:00Z') + pd.Timedelta(days=i),
                              score_gap=(i % 5) - 2,
                              **{f: ((i % 3) - 1) * .1 for f in STRUCT_GAPS}) for i in range(n)])


class SnapshotTests(unittest.TestCase):
    def test_overtime_retained_and_terminal_checkpoint_excluded(self):
        d, h = example()
        snap = build_snapshots(d, pd.DataFrame([h]), checkpoints=(24, 27, 30))
        self.assertEqual(snap.k.tolist(), [24, 27])
        self.assertEqual(snap.score_gap.tolist(), [0, 1])
        self.assertEqual(snap.checkpoint.unique().tolist(), ['next_round_freeze_end'])

    def test_regulation_terminal_not_scored(self):
        d, h = example([A] * 12 + [B] * 5 + [A])
        snap = build_snapshots(d, pd.DataFrame([h]), checkpoints=(12, 18))
        self.assertEqual(snap.k.tolist(), [12])

    def test_overtime_moves_all_team_fields_with_roster(self):
        d, h = example()
        normalized = normalize_round_events(d, h)
        row = normalized[normalized.round_num == 28].iloc[0]
        self.assertEqual(row.ct_roster, A)
        self.assertEqual(row.ct_hp, 300)
        self.assertEqual(row.t_hp, 200)
        self.assertEqual(row.ct_trade, 1)
        self.assertEqual(row.winner_side, d.loc[27, 'winner_side'])
        self.assertEqual(row.winner_roster, B)

    def test_future_round_features_do_not_change_checkpoint(self):
        d, h = example()
        first = build_snapshots(d, pd.DataFrame([h]), checkpoints=(6,))
        d.loc[d.round_num > 6, ['ct_hp', 't_hp', 'ct_trade', 't_trade']] = 9999
        second = build_snapshots(d, pd.DataFrame([h]), checkpoints=(6,))
        pd.testing.assert_frame_equal(first, second)

    def test_same_series_maps_not_mixed(self):
        d1, h1 = example(map_no=1)
        d2, h2 = example([B] * 12 + [A] * 5 + [B], map_no=2)
        snap = build_snapshots(pd.concat([d1, d2]), pd.DataFrame([h1, h2]), checkpoints=(6,))
        self.assertEqual(len(snap), 2)
        self.assertEqual(sorted(snap.score_gap), [-6, 6])
        self.assertEqual(snap.match_id.nunique(), 1)

    def test_missing_round_is_rejected(self):
        d, h = example()
        snap = build_snapshots(d[d.round_num != 5], pd.DataFrame([h]))
        self.assertTrue(snap.empty)
        self.assertEqual(snap.attrs['audit']['rejected_maps']['non_contiguous_rounds'], 1)

    def test_history_label_conflict_is_rejected(self):
        d, h = example()
        h['y'] = 1 - h['y']
        snap = build_snapshots(d, pd.DataFrame([h]))
        self.assertTrue(snap.empty)
        self.assertEqual(snap.attrs['audit']['rejected_maps']['history_mismatch_y'], 1)

    def test_duplicate_clean_map_is_error(self):
        d, h = example()
        with self.assertRaisesRegex(ValueError, 'duplicate_clean_map'):
            build_snapshots(d, pd.DataFrame([h, h]))


class ChronologicalTests(unittest.TestCase):
    def test_unsafe_legacy_mapwin_entry_points_are_quarantined(self):
        from cs2ml import edge, mapwin_accumulate
        for module in (edge, mapwin_accumulate):
            with self.assertRaisesRegex(RuntimeError, 'legacy_mapwin_evaluation_disabled'):
                module._load()

    def test_no_future_or_unavailable_training_results(self):
        d = comparison_rows()
        d.loc[0, 'available_at'] = pd.Timestamp('2027-01-01', tz='UTC')
        for tr, te in forward_splits(d):
            self.assertNotIn(0, tr)
            self.assertLess(d.iloc[tr].available_at.max(), d.iloc[te].start_at.min())
            self.assertFalse(set(d.iloc[tr].match_id) & set(d.iloc[te].match_id))

    def test_same_time_series_and_sibling_maps_stay_together(self):
        d = comparison_rows()
        sibling = d.iloc[[7]].copy()
        sibling['map_id'] = 'sibling'
        d = pd.concat([d, sibling], ignore_index=True)
        d.loc[8, ['start_at', 'available_at']] = d.loc[7, ['start_at', 'available_at']].to_numpy()
        for tr, te in forward_splits(d):
            self.assertEqual(7 in te, 20 in te)
            self.assertEqual(7 in te, 8 in te)
            self.assertFalse((7 in tr) and (20 in te))

    def test_all_candidates_share_complete_cases_and_scored_maps(self):
        d = comparison_rows()
        d.loc[9, 'hp_gap'] = np.nan
        report, pred = compare_variants(d, min_train=2)
        self.assertEqual(report['common_complete_rows'], 19)
        self.assertNotIn('map-9', pred.map_id.tolist())
        self.assertEqual(len({m['n'] for m in report['metrics'].values()}), 1)
        self.assertGreater(report['scored_maps'], 0)
        for name in VARIANTS:
            self.assertTrue(pred[f'p_{name}'].notna().equals(pred.p_score_only.notna()))

    def test_future_label_changes_do_not_change_earlier_predictions(self):
        d = comparison_rows()
        _, p1 = compare_variants(d, min_train=2)
        d.loc[d.index >= 16, 'y'] = 1 - d.loc[d.index >= 16, 'y']
        _, p2 = compare_variants(d, min_train=2)
        cols = [f'p_{name}' for name in VARIANTS]
        pd.testing.assert_frame_equal(p1[cols], p2[cols])

    def test_series_metadata_disagreement_fails_closed(self):
        d = comparison_rows()
        d.loc[1, 'match_id'] = d.loc[0, 'match_id']
        with self.assertRaisesRegex(ValueError, 'inconsistent_series_metadata'):
            list(forward_splits(d))

    def test_empty_common_cohort_has_no_fabricated_score(self):
        d = comparison_rows()
        d['hp_gap'] = np.nan
        report, pred = compare_variants(d, min_train=2)
        self.assertEqual(report['scored_maps'], 0)
        self.assertTrue(pred.empty)


if __name__ == '__main__':
    unittest.main()
