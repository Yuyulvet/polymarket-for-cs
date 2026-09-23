"""As-of CS2 fundamental annotation for market shocks; never a trade signal."""
from __future__ import annotations

from datetime import datetime
import math

import pandas as pd

ANNOTATION_SCHEMA_VERSION = 1


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _timestamp(value):
    numeric = _number(value)
    if numeric is not None:
        return numeric
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _prediction_time(row: dict):
    for key in ("prediction_timestamp", "information_timestamp", "available_at",
                "received_at", "observed_at"):
        value = _timestamp(row.get(key))
        if value is not None:
            return value
    return None


def _identity_matches(shock: dict, prediction: dict) -> bool:
    compared = False
    for key in ("event_id", "market_id", "match_id"):
        left, right = shock.get(key), prediction.get(key)
        if left is not None and right is not None:
            compared = True
            if str(left) != str(right):
                return False
    if not compared:
        return False
    shock_map = shock.get("map_number")
    pred_map = prediction.get("map_number")
    if shock_map is not None and pred_map is not None and int(shock_map) != int(pred_map):
        return False
    shock_outcome = str(shock.get("outcome") or "").casefold()
    pred_outcome = str(prediction.get("outcome") or prediction.get("team")
                       or prediction.get("focal_outcome") or "").casefold()
    return not (shock_outcome and pred_outcome) or shock_outcome == pred_outcome


def _probability(row: dict):
    for key in ("probability", "pre_match_probability", "p_lgbm", "p_lr", "p_glicko"):
        value = _number(row.get(key))
        if value is not None and 0 <= value <= 1:
            return value
    return None


def annotate_shocks(shocks: list[dict], predictions: pd.DataFrame | list[dict]) -> list[dict]:
    prediction_rows = (predictions.to_dict("records") if isinstance(predictions, pd.DataFrame)
                       else list(predictions))
    normalized = []
    for row in prediction_rows:
        if not isinstance(row, dict):
            continue
        timestamp = _prediction_time(row)
        probability = _probability(row)
        if timestamp is not None and probability is not None:
            normalized.append((timestamp, probability, row))
    normalized.sort(key=lambda item: item[0])

    annotations = []
    for shock in shocks:
        decision = _number(shock.get("decision_ts"))
        eligible = [(timestamp, probability, row)
                    for timestamp, probability, row in normalized
                    if decision is not None and timestamp <= decision
                    and _identity_matches(shock, row)]
        base = {
            "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
            "shock_id": shock.get("shock_id"), "token": shock.get("token"),
            "annotation_role": "descriptive_only_not_trading_input",
        }
        if not eligible:
            base.update({
                "pre_match_cs2_probability": None, "cs2_model_name": None,
                "cs2_model_version": None, "cs2_uncertainty": None,
                "cs2_prediction_timestamp": None, "cs2_information_age_seconds": None,
                "cs2_map": None, "cs2_team_identity": None,
                "roster_confidence": None, "tier": None, "history_coverage": None,
                "cs2_edge_before_shock": None, "shock_alignment": None,
                "annotation_available": False,
                "missing_reason": "no_identity_matched_prediction_available_as_of_decision",
            })
            annotations.append(base)
            continue
        timestamp, probability, row = max(eligible, key=lambda item: item[0])
        prior = _number(shock.get("pre_shock_mid"))
        edge = probability - prior if prior is not None else None
        shock_change = _number(shock.get("shock_change"))
        alignment = None
        if edge is not None and shock_change is not None:
            if edge == 0 or shock_change == 0:
                alignment = "neutral"
            else:
                alignment = "aligned" if edge * shock_change > 0 else "opposed"
        base.update({
            "pre_match_cs2_probability": probability,
            "cs2_model_name": row.get("model_name") or row.get("model"),
            "cs2_model_version": row.get("model_version") or row.get("version"),
            "cs2_uncertainty": _number(row.get("uncertainty")),
            "cs2_prediction_timestamp": timestamp,
            "cs2_information_age_seconds": decision - timestamp,
            "cs2_map": row.get("map_name") or row.get("map"),
            "cs2_team_identity": row.get("outcome") or row.get("team")
            or row.get("focal_outcome"),
            "roster_confidence": row.get("roster_confidence"),
            "tier": row.get("tier"), "history_coverage": row.get("history_coverage"),
            "cs2_edge_before_shock": edge, "shock_alignment": alignment,
            "annotation_available": True, "missing_reason": None,
        })
        annotations.append(base)
    return annotations
