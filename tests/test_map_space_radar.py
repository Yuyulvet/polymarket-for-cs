import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from cs2ml.map_space import MAPS, build_catalog, fit_calibration, project_pixels
from cs2ml.stream_radar_capture import EvidenceWriter, radar_box, choose_format, capture


def spec():
    def point(x, y):
        return {'pixel': [x, y], 'world_xy': [2*x+10, -3*y+100]}
    return {'map_name': 'de_cache', 'client_version': '2000908', 'frame_size': [200, 200],
            'max_error_world_units': 1, 'anchors': [point(10, 10), point(100, 10), point(10, 100)],
            'checkpoints': [point(30, 30), point(90, 80)]}


class MapSpaceTests(unittest.TestCase):
    def test_all_seven_and_missing(self):
        m = {'client_version': 2000908, 'maps': list(MAPS)}
        d = {name: {'pos_x': 0, 'pos_y': 0, 'scale': 5} for name in MAPS}
        self.assertEqual(len(build_catalog(m, d, '2000908')['maps']), 7)
        del d['de_cache']
        with self.assertRaisesRegex(ValueError, 'missing_map'):
            build_catalog(m, d, '2000908')

    def test_fit_round_trip_and_z_unknown(self):
        c = fit_calibration(spec())
        np.testing.assert_allclose(project_pixels(c, [[50, 50]], [200, 200], 'de_cache', '2000908'), [[110, -50]])
        self.assertIsNone(c['world_z'])
        self.assertFalse(c['eligible_for_inference'])

    def test_wrong_context(self):
        c = fit_calibration(spec())
        for size, name, version in (([100, 100], 'de_cache', '2000908'), ([200, 200], 'de_nuke', '2000908'),
                                    ([200, 200], 'de_cache', '1')):
            with self.assertRaisesRegex(ValueError, 'context'):
                project_pixels(c, [[50, 50]], size, name, version)

    def test_bad_landmarks(self):
        s = spec()
        s['checkpoints'][0]['world_xy'][0] += 100
        with self.assertRaisesRegex(ValueError, 'error_exceeds'):
            fit_calibration(s)
        s = spec()
        s['checkpoints'][0] = copy.deepcopy(s['anchors'][0])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            fit_calibration(s)
        s = spec()
        s['anchors'] = [{'pixel': [i, i], 'world_xy': [i, i]} for i in (10, 20, 40)]
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            fit_calibration(s)


class RadarCaptureTests(unittest.TestCase):
    def test_roi(self):
        self.assertEqual(radar_box((100, 100), '0,0,0.5,0.5'), (0, 0, 50, 50))
        for roi in ('nan,0,.2,.2', '-.1,0,.2,.2', '.9,0,.2,.2', '0,0,0,1', '1,2'):
            with self.assertRaises(ValueError):
                radar_box((100, 100), roi)

    def test_high_resolution_no_drm(self):
        fs = [{'url': 'a', 'vcodec': 'h264', 'height': h} for h in (480, 720, 1080)]
        self.assertEqual(choose_format(fs)['height'], 1080)
        with self.assertRaises(ValueError):
            choose_format(fs[:1])
        with self.assertRaises(ValueError):
            choose_format([{**fs[-1], 'has_drm': True}])

    def writer(self, directory, **kwargs):
        return EvidenceWriter(Path(directory)/'capture',
            {'event_id': '123', 'team_a': 'A', 'team_b': 'B', 'map_number': 1},
            'de_cache', '0,0,.5,.5', {'kind': 'synthetic_test'}, **kwargs)

    def test_evidence_not_inference_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            w = self.writer(td)
            row = w.save(Image.new('RGB', (100, 100)), pts_seconds=0., decoded_at='2026-09-18T00:00:00Z', decoded_monotonic_ns=1)
            self.assertIsNone(row['source_timestamp'])
            self.assertIsNone(row['player_observations'])
            self.assertIsNone(row['network_received_at'])
            self.assertTrue(all((w.output/f).exists() for f in row['files']))
            log = json.loads((w.output/'events.jsonl').read_text())
            self.assertFalse(log['eligible_for_inference'])
            with self.assertRaises(FileExistsError):
                self.writer(td)
            with self.assertRaisesRegex(ValueError, 'frame_size_changed'):
                w.save(Image.new('RGB', (200, 200)), pts_seconds=1., decoded_at='x', decoded_monotonic_ns=2)

    def test_budget_stops_before_writes(self):
        with tempfile.TemporaryDirectory() as td:
            w = self.writer(td, max_bytes=1)
            with self.assertRaisesRegex(InterruptedError, 'budget'):
                w.save(Image.new('RGB', (100, 100)), pts_seconds=0., decoded_at='x', decoded_monotonic_ns=1)
            self.assertEqual(list((w.output/'frames').iterdir()), [])

    def test_local_video_decode(self):
        import av
        with tempfile.TemporaryDirectory() as td:
            video = Path(td)/'fixture.mp4'
            with av.open(str(video), mode='w') as out:
                stream = out.add_stream('mpeg4', rate=10)
                stream.width = stream.height = 100
                stream.pix_fmt = 'yuv420p'
                for i in range(12):
                    frame = av.VideoFrame.from_image(Image.new('RGB', (100, 100), (i*10, 20, 30)))
                    for packet in stream.encode(frame):
                        out.mux(packet)
                for packet in stream.encode():
                    out.mux(packet)
            w = self.writer(td)
            capture(video, w, seconds=10, sample_hz=5, max_frames=3)
            rows = [json.loads(line) for line in (w.output/'events.jsonl').read_text().splitlines()]
            self.assertEqual(w.count, 3)
            self.assertEqual(rows[-1]['reason'], 'frame_limit')
            self.assertEqual([round(r['video_pts_seconds'], 2) for r in rows if r['type']=='radar_frame'], [0, .2, .4])


if __name__ == '__main__':
    unittest.main()
