"""Append-only observations and restart-safe local paper-desk state."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class DeskStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as con:
            con.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY, body TEXT NOT NULL, summary TEXT NOT NULL,
                    first_seen TEXT NOT NULL, last_seen TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS contexts (
                    event_id TEXT PRIMARY KEY, body TEXT NOT NULL, binding TEXT NOT NULL,
                    confirmed_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY, event_id TEXT NOT NULL, received_at TEXT NOT NULL,
                    body TEXT NOT NULL, UNIQUE(event_id, received_at));
                CREATE TABLE IF NOT EXISTS resolutions (
                    event_id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS resolution_checks (
                    event_id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, at TEXT NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS hltv_evidence (
                    id INTEGER PRIMARY KEY, event_id TEXT NOT NULL, body TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS hltv_evidence_event ON hltv_evidence(event_id, id);
                CREATE TABLE IF NOT EXISTS hltv_identities (
                    hltv_player_id TEXT PRIMARY KEY, steamid TEXT NOT NULL UNIQUE, body TEXT NOT NULL);
            """)

    @contextmanager
    def connection(self):
        con = sqlite3.connect(self.path, timeout=15)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    def upsert_events(self, events, summaries, at):
        with self.connection() as con:
            for event, summary in zip(events, summaries):
                con.execute("""INSERT INTO events VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO UPDATE SET body=excluded.body,
                    summary=excluded.summary, last_seen=excluded.last_seen""",
                            (summary["event_id"], encode(event), encode(summary), at, at))

    def events(self):
        with self.connection() as con:
            return [{"event": json.loads(r["body"]), "summary": json.loads(r["summary"]),
                     "first_seen": r["first_seen"], "last_seen": r["last_seen"]}
                    for r in con.execute("SELECT * FROM events ORDER BY event_id")]

    def save_context(self, context, binding, at):
        eid = str(context["event_id"])
        with self.connection() as con:
            # A recorded forecast must never be silently relabelled by a new roster/map.
            if con.execute("SELECT 1 FROM snapshots WHERE event_id=? LIMIT 1", (eid,)).fetchone():
                raise ValueError("context_locked_after_first_observation")
            con.execute("INSERT INTO contexts VALUES (?, ?, ?, ?) ON CONFLICT(event_id) DO UPDATE SET "
                        "body=excluded.body, binding=excluded.binding, confirmed_at=excluded.confirmed_at",
                        (eid, encode(context), encode(binding), at))
            con.execute("INSERT INTO runs(kind, at, body) VALUES ('confirmation', ?, ?)",
                        (at, encode({"event_id": eid, "context": context, "binding": binding})))

    def contexts(self):
        with self.connection() as con:
            return {r["event_id"]: {"context": json.loads(r["body"]),
                                    "binding": json.loads(r["binding"]), "confirmed_at": r["confirmed_at"]}
                    for r in con.execute("SELECT * FROM contexts")}

    def add_snapshot(self, snapshot):
        with self.connection() as con:
            existing = con.execute("SELECT body FROM snapshots WHERE event_id=? AND received_at=?",
                                   (str(snapshot["event_id"]), snapshot["received_at"])).fetchone()
            if existing and json.loads(existing[0]) != snapshot:
                raise ValueError("conflicting_observation_at_same_timestamp")
            con.execute("INSERT OR IGNORE INTO snapshots(event_id, received_at, body) VALUES (?, ?, ?)",
                        (str(snapshot["event_id"]), snapshot["received_at"], encode(snapshot)))

    def snapshots(self):
        with self.connection() as con:
            return [json.loads(r[0]) for r in con.execute("SELECT body FROM snapshots ORDER BY id")]

    def add_resolution(self, resolution):
        eid = str(resolution["event_id"])
        with self.connection() as con:
            old = con.execute("SELECT body FROM resolutions WHERE event_id=?", (eid,)).fetchone()
            if old:
                old = json.loads(old[0])
                if any(old[k] != resolution[k] for k in ("condition_id", "payouts")):
                    raise ValueError("conflicting_resolution_requires_review")
                return False
            con.execute("INSERT INTO resolutions VALUES (?, ?)", (eid, encode(resolution)))
            return True

    def resolutions(self):
        with self.connection() as con:
            return [json.loads(r[0]) for r in con.execute("SELECT body FROM resolutions ORDER BY event_id")]

    def check_resolution(self, event_id, status, at, reason=None):
        with self.connection() as con:
            con.execute("INSERT INTO resolution_checks VALUES (?, ?) ON CONFLICT(event_id) DO UPDATE SET body=excluded.body",
                        (str(event_id), encode({"status": status, "checked_at": at, "reason": reason})))

    def resolution_checks(self):
        with self.connection() as con:
            return {r["event_id"]: json.loads(r["body"]) for r in con.execute("SELECT * FROM resolution_checks")}

    def log(self, kind, at, result):
        with self.connection() as con:
            con.execute("INSERT INTO runs(kind, at, body) VALUES (?, ?, ?)", (kind, at, encode(result)))

    def logs(self, limit=30):
        with self.connection() as con:
            return [{"kind": r["kind"], "at": r["at"], "result": json.loads(r["body"])}
                    for r in con.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))]

    def pin_protocol(self, protocol):
        with self.connection() as con:
            old = con.execute("SELECT body FROM metadata WHERE key='protocol'").fetchone()
            if old and json.loads(old[0]) != protocol:
                if any(con.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
                       for table in ("contexts", "snapshots", "resolutions")):
                    raise ValueError("paper_protocol_changed_use_a_new_database")
                # Discovery-only databases have no research cohort to relabel.
            con.execute("INSERT INTO metadata VALUES ('protocol', ?) ON CONFLICT(key) DO UPDATE SET body=excluded.body",
                        (encode(protocol),))

    def add_hltv_evidence(self, evidence):
        with self.connection() as con:
            cursor = con.execute("INSERT INTO hltv_evidence(event_id, body) VALUES (?, ?)",
                                 (str(evidence["event_id"]), encode(evidence)))
            return cursor.lastrowid

    def hltv_evidence(self, evidence_id=None):
        with self.connection() as con:
            if evidence_id is not None:
                row = con.execute("SELECT * FROM hltv_evidence WHERE id=?", (int(evidence_id),)).fetchone()
                if not row:
                    raise ValueError("hltv_evidence_not_found")
                return {**json.loads(row["body"]), "evidence_id": row["id"]}
            return [{**json.loads(r["body"]), "evidence_id": r["id"]}
                    for r in con.execute("SELECT * FROM hltv_evidence ORDER BY id")]

    def latest_hltv_evidence(self):
        with self.connection() as con:
            return {r["event_id"]: {**json.loads(r["body"]), "evidence_id": r["id"]}
                    for r in con.execute("SELECT * FROM hltv_evidence WHERE id IN "
                                         "(SELECT MAX(id) FROM hltv_evidence GROUP BY event_id)")}

    def hltv_identities(self):
        with self.connection() as con:
            return {r["hltv_player_id"]: json.loads(r["body"])
                    for r in con.execute("SELECT * FROM hltv_identities")}

    def save_hltv_identities(self, mappings):
        # The entire reviewed batch is atomic. A conflict never silently
        # rewrites identity history or partially applies the other rows.
        with self.connection() as con:
            for mapping in mappings:
                pid, steamid = str(mapping["hltv_player_id"]), str(mapping["steamid"])
                old = con.execute("SELECT steamid FROM hltv_identities WHERE hltv_player_id=?", (pid,)).fetchone()
                other = con.execute("SELECT hltv_player_id FROM hltv_identities WHERE steamid=?", (steamid,)).fetchone()
                if (old and old[0] != steamid) or (other and other[0] != pid):
                    raise ValueError("hltv_identity_conflict_requires_review")
                con.execute("INSERT OR IGNORE INTO hltv_identities VALUES (?, ?, ?)", (pid, steamid, encode(mapping)))
