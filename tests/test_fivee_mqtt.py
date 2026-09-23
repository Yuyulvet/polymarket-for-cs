from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cs2ml.fivee_mqtt as mod
from cs2ml.fivee_mqtt import source_version_timestamp, summarize_mqtt_message, topic_for


def player(name="p"):
    return {"id": name, "name": name, "hp": "100", "money": "800",
            "weapon": "glock", "helmet": "2", "kevlar": "2",
            "has_defusekit": "2", "kill": "0", "death": "0", "assist": "0"}


def message(match_id="m"):
    return {
        "event_name": "csgo-detail",
        "data": {"from_ver": "10", "this_ver": "11", "match": {
            "mc_info": {"id": match_id,
                        "t1_info": {"id": "a", "disp_name": "Alpha"},
                        "t2_info": {"id": "b", "disp_name": "Beta"}},
            "global_state": {"status": "1", "t1_score": "0", "t2_score": "0"},
            "bouts_state": [{"status": "1", "bout_num": "1", "map_name": "Nuke",
                             "curr_round_num": "1", "curr_bout_stage": "fh",
                             "t1_stats": {"all_score": "0", "fh_role": "CT"},
                             "t2_stats": {"all_score": "0", "fh_role": "T"},
                             "t1_pr_stats": [player("a1")],
                             "t2_pr_stats": [player("b1")]}],
        }},
    }


class FiveEMqttTests(unittest.TestCase):
    def test_topic(self):
        self.assertEqual(topic_for("abc"), "csgo/product/detail/abc")
        self.assertEqual(source_version_timestamp("00017896574061"), 1789657406.1)
        self.assertIsNone(source_version_timestamp("not-a-version"))

    def test_message_is_normalized_and_versions_preserved(self):
        summary, raw, versions = summarize_mqtt_message(message(), "m")
        self.assertEqual(summary["match_id"], "m")
        self.assertEqual(summary["live_bouts"][0]["team1"]["economy"]["money_sum"], 800)
        self.assertEqual(raw["mc_info"]["id"], "m")
        self.assertEqual(versions, {"from_ver": "10", "this_ver": "11"})

    def test_wrong_event_and_match_are_rejected(self):
        wrong = message()
        wrong["event_name"] = "csgo-agenda"
        with self.assertRaisesRegex(ValueError, "not_csgo_detail"):
            summarize_mqtt_message(wrong, "m")
        with self.assertRaisesRegex(ValueError, "match_id_mismatch"):
            summarize_mqtt_message(message("other"), "m")


class FakeReason:
    is_failure = False


class FakeClient:
    """Minimal paho stand-in: connect() immediately CONNACKs success."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.on_connect = None
        FakeClient.instances.append(self)

    def username_pw_set(self, username, password):
        self.username, self.password = username, password

    def ws_set_options(self, path):
        pass

    def tls_set(self, cert_reqs):
        pass

    def connect(self, host, port, keepalive):
        self.on_connect(self, None, None, FakeReason(), None)

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def subscribe(self, topic, qos=0):
        return (0, 1)


class CredentialRotationTests(unittest.TestCase):
    def test_each_connection_fetches_fresh_credentials(self):
        cred_calls = {"n": 0}

        def fake_post(url, **kwargs):
            cred_calls["n"] += 1
            response = mock.Mock()
            response.raise_for_status = lambda: None
            response.json = lambda: {"success": True, "data": {
                "client_id": f"client-{cred_calls['n']}",
                "username": f"user-{cred_calls['n']}",
                "password": "pw"}}
            return response

        def fake_get(url, **kwargs):
            response = mock.Mock()
            response.raise_for_status = lambda: None
            response.json = lambda: {"success": True,
                                     "data": {"match": message("m")["data"]["match"]}}
            return response

        FakeClient.instances = []
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(mod.mqtt, "Client", FakeClient), \
                mock.patch.object(mod, "ROTATE_SECONDS", 0.4):
            recorder = mod.FiveEMqttRecorder(
                "m", Path(directory) / "states.jsonl",
                duration_seconds=2.0, request_post=fake_post, request_get=fake_get)
            code = recorder.run()
            rows = [json.loads(line) for line in
                    (Path(directory) / "states.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(code, 0)
        connections = recorder._connection_count
        self.assertGreaterEqual(connections, 3)  # rotated before the 60-min cap
        self.assertEqual(cred_calls["n"], connections)  # fresh creds per connection
        client_ids = {c.kwargs["client_id"] for c in FakeClient.instances}
        self.assertEqual(len(client_ids), connections)
        # reconnect_on_failure must be off: stale-credential auto-reconnect is the bug
        for instance in FakeClient.instances:
            self.assertFalse(instance.kwargs.get("reconnect_on_failure", True))
        self.assertEqual(rows[-1]["record_type"], "session_end")
        self.assertEqual(rows[-1]["exit_reason"], "deadline")


if __name__ == "__main__":
    unittest.main()
