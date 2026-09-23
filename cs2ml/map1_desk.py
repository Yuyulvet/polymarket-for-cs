"""Local, paper-only orchestration. No credentials, wallet or order client."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
from pathlib import Path
import threading
import time

import pandas as pd

from .map1 import capture_once, now
from .map1_data import utc
from .map1_hltv_desk import HltvReview, MAX_EVIDENCE_AGE, MIN_REQUEST_INTERVAL, REFRESH_SECONDS
from .map1_market import bind_market, fetch_event, select_map1_market, validate_context
from .map1_paper import replay
from .map1_sources import CLOB, binary_resolution, describe_event, discover, public_get
from .map1_store import DeskStore

PAPER_PARAMETERS = {"initial_cash": 1000, "shares": 10, "min_edge": .05,
                    "latency_seconds": .5, "max_age": 10, "fill_window_seconds": 10}


def fingerprint(path):
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PaperDesk:
    def __init__(self, data_dir: Path, database: Path | None = None):
        self.data_dir = Path(data_dir)
        self.store = DeskStore(database or self.data_dir / "desk.sqlite")
        self.protocol = {
            "version": 2, "mode": "paper_only", "live_trading_enabled": False,
            "parameters": PAPER_PARAMETERS, "capture_window_minutes": 60,
            "hltv_evidence_max_age_seconds": MAX_EVIDENCE_AGE,
            "data_hashes": {n: fingerprint(self.data_dir / n)
                            for n in ("history.parquet", "features.parquet")},
            "code_hashes": {n: fingerprint(Path(__file__).with_name(n)) for n in
                            ("map1.py", "map1_data.py", "map1_model.py", "map1_market.py", "map1_paper.py",
                             "map1_sources.py", "map1_desk.py", "map1_store.py",
                             "map1_hltv.py", "map1_hltv_desk.py", "map1_eligibility.py")}}
        self.store.pin_protocol(self.protocol)
        self.hltv = HltvReview(self.store, self.data_dir)
        self.operation_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="map1-desk")
        self.future = None
        self.active_job = None
        self.running = False  # Never resume network work without a new explicit Start.
        self.closed = threading.Event()
        self.last_discovery = 0.0
        self.last_settlement = 0.0
        self.history = self.features = None
        self.data_error = None
        self.report_cache = None
        self.report_version = 0
        try:
            self.history = pd.read_parquet(self.data_dir / "history.parquet")
            self.features = pd.read_parquet(self.data_dir / "features.parquet")
        except Exception as exc:
            self.data_error = f"research_data_unavailable: {type(exc).__name__}: {exc}"
        self.scheduler = threading.Thread(target=self._schedule, daemon=True, name="map1-scheduler")
        self.scheduler.start()

    def close(self):
        self.running = False
        self.closed.set()
        self.scheduler.join(timeout=2)
        self.executor.shutdown(wait=True, cancel_futures=True)

    def _invalidate(self):
        with self.state_lock:
            self.report_version += 1
            self.report_cache = None

    def _schedule(self):
        # Interruptible stop; no hidden OS service, cron job or auto-start entry.
        while not self.closed.wait(3):
            if self.running:
                try:
                    self.submit("cycle")
                except ValueError:
                    pass  # A previous bounded round is still working.

    def submit(self, name, payload=None):
        if name not in {"discover", "capture", "settle", "confirm", "cycle",
                        "hltv_import", "hltv_identities", "hltv_confirm"}:
            raise ValueError("unknown_operation")
        with self.state_lock:
            if self.closed.is_set():
                raise ValueError("desk_closed")
            if self.future is not None and not self.future.done():
                raise ValueError("another_operation_is_running")
            self.active_job = name
            self.future = self.executor.submit(self.run, name, payload)
        return {"accepted": True, "operation": name}

    def run(self, name, payload=None):
        with self.operation_lock:
            started = now()
            try:
                if name == "cycle":
                    result = self._cycle()
                else:
                    method = {"discover": self.discover_once, "capture": self.capture_round,
                              "settle": self.settle_once, "confirm": self.confirm,
                              "hltv_import": self.import_hltv,
                              "hltv_identities": self.hltv.save_identities,
                              "hltv_confirm": self.confirm_hltv}[name]
                    result = method(payload) if name in {"confirm", "hltv_import", "hltv_identities", "hltv_confirm"} else method()
                self.store.log(name, started, {"status": "ok", "finished_at": now(), **result})
                return result
            except Exception as exc:
                result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}", "finished_at": now()}
                self.store.log(name, started, result)
                return result
            finally:
                self._invalidate()
                with self.state_lock:
                    self.active_job = None

    def _cycle(self):
        result = {}
        # Each subsystem fails independently. Discovery failures must not stop
        # recording already-confirmed events or resolving existing positions.
        jobs = []
        current = time.monotonic()
        if current - self.last_discovery >= 300:
            self.last_discovery = current
            jobs.append(("discovery", self.discover_once))
        jobs.append(("hltv_refresh", self.refresh_hltv_once))
        jobs.append(("capture", self.capture_round))
        if current - self.last_settlement >= 60:
            self.last_settlement = current
            jobs.append(("settlement", self.settle_once))
        for key, job in jobs:
            if not self.running or self.closed.is_set():
                break
            try:
                result[key] = job()
            except Exception as exc:
                result[key] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        return result

    def discover_once(self):
        result = discover(now())
        at = now()
        summaries = [describe_event(event, at) for event in result["events"]]
        self.store.upsert_events(result["events"], summaries, at)
        return {"seen": len(summaries), "eligible": sum(s["status"] == "awaiting_confirmation" for s in summaries),
                "truncated": result["truncated"], "pages": result["pages"],
                "reasons": dict(Counter(s["reason"] for s in summaries)),
                "source": result["sport_metadata"],
                "warning": "discovery_page_limit_reached" if result["truncated"] else None}

    def confirm(self, submitted):
        if not isinstance(submitted, dict):
            raise ValueError("context_must_be_an_object")
        context = copy.deepcopy(submitted)
        if any(str(k).startswith("hltv_") for k in context):
            raise ValueError("manual_context_cannot_claim_hltv_verification")
        # Validate claimed publication times, but never use them to backdate our
        # own information receipt. This desk knows the evidence only from now.
        at = now()
        validate_context(context, at)
        event = fetch_event(str(context["event_id"]))
        received = now()
        summary = describe_event(event, received)
        if summary["status"] != "awaiting_confirmation":
            raise ValueError(summary["reason"])
        context["declared_map_known_at"] = context["map_known_at"]
        context["declared_roster_known_at"] = context["roster_known_at"]
        context["map_known_at"] = context["roster_known_at"] = received
        context["confirmed_at"] = received
        context["confirmation_method"] = "manual_source_attestation_not_automated_verification"
        binding = bind_market(event, context, received)
        self.store.save_context(context, binding, received)
        self.store.upsert_events([event], [summary], received)
        return {"event_id": str(context["event_id"]), "confirmed_at": received}

    def import_hltv(self, submitted):
        if not isinstance(submitted, dict):
            raise ValueError("request_must_be_object")
        # Validate the URL before any network call, including market metadata.
        from .map1_hltv import normalize_match_url
        url = normalize_match_url(submitted["url"])
        event = fetch_event(str(submitted["event_id"]))
        if str(event.get("id")) != str(submitted["event_id"]):
            raise ValueError("event_id_mismatch")
        result = self.hltv.import_match(event, url)
        self.store.upsert_events([event], [describe_event(event, now())], now())
        return {"event_id": str(event["id"]), "evidence_id": result["evidence_id"],
                "ready_for_review": result["ready"], "issues": result["issues"], "error": result["error"]}

    def confirm_hltv(self, submitted):
        if not isinstance(submitted, dict):
            raise ValueError("request_must_be_object")
        event = fetch_event(str(submitted["event_id"]))
        if str(event.get("id")) != str(submitted["event_id"]):
            raise ValueError("event_id_mismatch")
        received = now()
        context = self.hltv.confirmed_context(submitted, event, received)
        binding = bind_market(event, context, received)
        self.store.save_context(context, binding, received)
        self.store.upsert_events([event], [describe_event(event, received)], received)
        return {"event_id": str(event["id"]), "confirmed_at": received,
                "evidence_id": context["hltv_evidence_id"]}

    def refresh_hltv_once(self):
        # At most one source request per cycle; globally spaced >=15 seconds.
        # Refresh ONLY explicitly reviewed matches within the capture window.
        # Access/parse failures stop automatic retries until a manual import.
        if (self.hltv.last_request is not None and
                time.monotonic() - self.hltv.last_request < MIN_REQUEST_INTERVAL):
            return {"refreshed": 0, "reason": "hltv_request_cooldown"}
        at = now()
        evidence = self.store.latest_hltv_evidence()
        due = []
        for eid, saved in self.store.contexts().items():
            context = saved["context"]
            latest = evidence.get(eid)
            if (not context.get("hltv_match_id") or not latest or latest.get("error")
                    or latest.get("parsed", {}).get("issues")):
                continue
            start = utc(context["scheduled_start_at"])
            if not utc(at) < start <= utc(at) + pd.Timedelta(minutes=60):
                continue
            if (utc(at) - utc(latest["received_at"])).total_seconds() >= REFRESH_SECONDS:
                due.append((latest["received_at"], eid, latest["url"]))
        if not due:
            return {"refreshed": 0}
        _, eid, url = min(due)
        return {"refreshed": 1, **self.import_hltv({"event_id": eid, "url": url})}

    def capture_round(self):
        at = now()
        resolved = {str(r["event_id"]) for r in self.store.resolutions()}
        due = []
        for eid, saved in self.store.contexts().items():
            start = utc(saved["context"]["scheduled_start_at"])
            if eid not in resolved and utc(at) < start <= utc(at) + pd.Timedelta(minutes=60):
                due.append(saved["context"])
        if not due:
            return {"captured": 0, "reason": "no_confirmed_events_in_capture_window"}
        if len(due) > 20:
            raise ValueError("too_many_simultaneous_events_capture_capacity_exceeded")

        def capture(context):
            receipt = None
            if context.get("hltv_match_id"):
                try:
                    receipt = self.hltv.capture_guard(context, now())
                except ValueError as exc:
                    return {"schema_version": 1, "event_id": str(context["event_id"]), "context": context,
                            "received_at": now(), "status": "blocked", "reason": str(exc),
                            "model_target": "map1_winner"}
            if self.data_error:
                return {"schema_version": 1, "event_id": str(context["event_id"]), "context": context,
                        "received_at": now(), "status": "blocked", "reason": self.data_error,
                        "model_target": "map1_winner"}
            result = capture_once(context, self.history, self.features)
            if receipt:
                result["hltv_source_evidence"] = receipt
                # Long-running requests must not carry evidence past its expiry.
                try:
                    self.hltv.capture_guard(context, result["received_at"])
                except ValueError as exc:
                    result.update(status="blocked", reason=str(exc))
                    result.pop("decision", None)
            return result

        with ThreadPoolExecutor(max_workers=4) as pool:
            snapshots = list(pool.map(capture, due))
        for snapshot in snapshots:
            snapshot["protocol_version"] = self.protocol["version"]
            self.store.add_snapshot(snapshot)
        return {"captured": len(snapshots), "ready": sum(s["status"] == "ready" for s in snapshots),
                "reasons": dict(Counter(s.get("reason", "ready") for s in snapshots))}

    def settle_once(self):
        settled = {str(r["event_id"]) for r in self.store.resolutions()}
        added, waiting, errors = [], [], {}
        for eid, saved in self.store.contexts().items():
            if eid in settled or utc(saved["context"]["scheduled_start_at"]) > utc(now()):
                continue
            try:
                event = fetch_event(eid)
                market = select_map1_market(event)
                if market.get("closed") is not True or market.get("umaResolutionStatus") != "resolved":
                    waiting.append(eid)
                    self.store.check_resolution(eid, "awaiting_official_resolution", now())
                    continue
                binding = saved["binding"]
                clob = public_get(CLOB + "/markets/" + binding["condition_id"])
                result = binary_resolution(event, clob, binding, now())
                if result is None:
                    waiting.append(eid)
                    self.store.check_resolution(eid, "awaiting_official_resolution", now())
                elif self.store.add_resolution(result):
                    added.append(eid)
                    self.store.check_resolution(eid, "resolved", now())
            except Exception as exc:
                errors[eid] = f"{type(exc).__name__}: {exc}"
                self.store.check_resolution(eid, "requires_review", now(), errors[eid])
        return {"settled": added, "waiting": waiting, "requires_review": errors}

    def report(self):
        with self.state_lock:
            version = self.report_version
            if self.report_cache and time.monotonic() - self.report_cache[0] < 3:
                return self.report_cache[1]
        at = now()
        snapshots = self.store.snapshots()
        resolutions = self.store.resolutions()
        result = replay(snapshots, resolutions, **PAPER_PARAMETERS, as_of=at)
        result["as_of"] = at
        result["protocol"] = self.protocol
        result["snapshot_count"] = len(snapshots)
        result["observed_events"] = len({s["event_id"] for s in snapshots})
        result["resolved_events"] = len(resolutions)
        result["fill_count"] = sum(x["type"] == "fill" for x in result["ledger"])
        result["settled_trade_count"] = sum(x["type"] == "settle" for x in result["ledger"])
        result["skip_reasons"] = dict(Counter(x["reason"] for x in result["ledger"] if "reason" in x))
        latest = {}
        for snap in sorted(snapshots, key=lambda s: utc(s["received_at"])):
            signal = snap.get("decision") or {"action": "skip", "reason": snap.get("reason", "not_ready")}
            latest[snap["event_id"]] = {"status": snap["status"], "received_at": snap["received_at"],
                                       "decision": signal, "coverage": snap.get("coverage"),
                                       "p_model": snap.get("p_model")}
        result["latest"] = latest
        result["live_readiness"] = "NOT_AUTHORIZED_NOT_VALIDATED"
        with self.state_lock:
            if self.report_version == version:
                self.report_cache = (time.monotonic(), result)
        return result

    def state(self):
        at = now()
        confirmed = self.store.contexts()
        report = self.report()
        settled = {str(r["event_id"]) for r in self.store.resolutions()}
        resolution_checks = self.store.resolution_checks()
        hltv_evidence = self.store.latest_hltv_evidence()
        hltv_identities = self.store.hltv_identities()
        entries = []
        for row in self.store.events():
            summary = describe_event(row["event"], at)
            eid = summary["event_id"]
            summary.update(first_seen=row["first_seen"], last_seen=row["last_seen"],
                           confirmed=eid in confirmed, settled=eid in settled,
                           settlement_check=resolution_checks.get(eid),
                           latest=report["latest"].get(eid))
            if eid in confirmed:
                summary["context"] = confirmed[eid]["context"]
            if eid in hltv_evidence:
                summary["hltv"] = self.hltv.preview(hltv_evidence[eid], row["event"], at, hltv_identities)
            entries.append(summary)
        entries.sort(key=lambda s: (s["status"] == "excluded", s.get("scheduled_start_at", "9999"), s["event_id"]))
        with self.state_lock:
            active_job = self.active_job
        # Keep the UI response small. Full raw observations are available via
        # explicit JSONL export; book payloads never go into HTML.
        compact = {k: v for k, v in report.items() if k not in {"ledger", "latest", "open_positions"}}
        compact["ledger"] = report["ledger"][-100:]
        compact["open_positions"] = [{k: p[k] for k in ("event_id", "token", "shares", "cash", "filled_at")}
                                     for p in report["open_positions"]]
        logs = self.store.logs()
        warnings = []
        for item in logs[:1]:
            result = item["result"]
            if result.get("warning"):
                warnings.append(result["warning"])
            for key, value in result.items():
                if isinstance(value, dict) and value.get("status") == "error":
                    warnings.append(f"{key}: {value.get('reason')}")
        reviews = sum(c["status"] == "requires_review" for c in resolution_checks.values())
        if reviews:
            warnings.append(f"{reviews} 场结算需要核验，未知结果未入账；勾选显示已过期 / 排除查看。")
        return {"mode": "paper_only", "live_trading_enabled": False, "running": self.running,
                "active_job": active_job, "as_of": at, "data_error": self.data_error,
                "events": entries, "report": compact, "runs": logs, "warnings": warnings,
                "database": str(self.store.path.resolve())}
