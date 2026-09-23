"""Deterministic descriptive shock taxonomy with a strict future-label split."""
from __future__ import annotations

import math

TAXONOMY_CONFIG = {
    "version": "shock_taxonomy_v1",
    "status": "exploratory_descriptive_only",
    "thin_book": {"max_l5_total_depth": 50.0, "max_window_volume": 10.0,
                  "max_trades_per_second": 1.0, "min_spread": 0.03},
    "aggressive_flow_candidate": {"min_trades_per_second": 2.0,
                                  "min_abs_trade_imbalance": 0.50,
                                  "min_same_side_depth_depletion": 0.25,
                                  "min_window_volume": 10.0},
    "broad_repricing": {"minimum_same_direction_quote_move": 0.0,
                        "min_trades_per_second": 1.0},
    "ex_post_reversal": {"negative_direction_adjusted_mid_drift": 0.0},
}

REALTIME_FIELDS = (
    "shock_direction", "shock_window_seconds", "spread", "liquidity",
    "trade_imbalance_1s", "trade_imbalance_2s", "trade_imbalance_5s",
    "volume_1s", "volume_2s", "volume_5s",
    "trades_per_second_1s", "trades_per_second_2s", "trades_per_second_5s",
    "bid_depth_depletion_1s", "bid_depth_depletion_2s", "bid_depth_depletion_5s",
    "ask_depth_depletion_1s", "ask_depth_depletion_2s", "ask_depth_depletion_5s",
    "pre_shock_bid", "pre_shock_ask", "post_shock_bid", "post_shock_ask",
    "pre_shock_obi_1", "pre_shock_obi_5", "obi_1", "obi_5",
)


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _window_value(shock: dict, prefix: str):
    window = int(float(shock.get("shock_window_seconds") or 1))
    return _number(shock.get(f"{prefix}_{window}s"))


def classify_realtime(shock: dict, config: dict = TAXONOMY_CONFIG) -> dict:
    """Use decision-time fields only; future returns are intentionally ignored."""
    direction = shock.get("shock_direction")
    spread, depth = _number(shock.get("spread")), _number(shock.get("liquidity"))
    volume = _window_value(shock, "volume")
    intensity = _window_value(shock, "trades_per_second")
    imbalance = _window_value(shock, "trade_imbalance")
    depletion_name = ("ask_depth_depletion" if direction == "up"
                      else "bid_depth_depletion")
    depletion = _window_value(shock, depletion_name)
    missing = [name for name, value in (("direction", direction), ("spread", spread),
               ("l5_total_depth", depth), ("volume", volume),
               ("trade_intensity", intensity)) if value is None]
    if missing:
        return {"taxonomy_version": config["version"],
                "realtime_class": "UNCLASSIFIED_INSUFFICIENT_DATA",
                "reasons": [f"missing_{name}" for name in missing]}

    thin = config["thin_book"]
    if (depth <= thin["max_l5_total_depth"]
            and volume <= thin["max_window_volume"]
            and intensity <= thin["max_trades_per_second"]
            and spread >= thin["min_spread"]):
        return {"taxonomy_version": config["version"],
                "realtime_class": "TYPE_2_THIN_BOOK_JUMP",
                "reasons": ["low_depth_low_flow_wide_spread"]}

    aggressive = config["aggressive_flow_candidate"]
    if (imbalance is not None and depletion is not None
            and intensity >= aggressive["min_trades_per_second"]
            and abs(imbalance) >= aggressive["min_abs_trade_imbalance"]
            and depletion >= aggressive["min_same_side_depth_depletion"]
            and volume >= aggressive["min_window_volume"]):
        return {"taxonomy_version": config["version"],
                "realtime_class": "TYPE_1_AGGRESSIVE_FLOW_CANDIDATE",
                "reasons": ["directional_flow_and_same_side_depth_depletion"]}

    pre_bid, pre_ask = _number(shock.get("pre_shock_bid")), _number(
        shock.get("pre_shock_ask"))
    post_bid, post_ask = _number(shock.get("post_shock_bid")), _number(
        shock.get("post_shock_ask"))
    signed = 1 if direction == "up" else -1
    broad = config["broad_repricing"]
    if (None not in (pre_bid, pre_ask, post_bid, post_ask)
            and signed * (post_bid - pre_bid) > broad["minimum_same_direction_quote_move"]
            and signed * (post_ask - pre_ask) > broad["minimum_same_direction_quote_move"]
            and intensity >= broad["min_trades_per_second"]):
        return {"taxonomy_version": config["version"],
                "realtime_class": "TYPE_3_BROAD_REPRICING",
                "reasons": ["bid_and_ask_moved_with_shock"]}
    return {"taxonomy_version": config["version"],
            "realtime_class": "OTHER_OR_MIXED",
            "reasons": ["no_v1_realtime_rule_matched"]}


def label_ex_post(observation: dict) -> dict:
    """Evaluation-only label; never a decision-time feature."""
    drift = _number(observation.get("raw_mid_drift"))
    if drift is None or not observation.get("label_valid"):
        label = "UNAVAILABLE"
    elif drift < TAXONOMY_CONFIG["ex_post_reversal"][
            "negative_direction_adjusted_mid_drift"]:
        label = "TYPE_4_TRANSIENT_OR_REVERSAL"
    elif drift > 0:
        label = "CONTINUATION"
    else:
        label = "FLAT"
    return {"taxonomy_version": TAXONOMY_CONFIG["version"],
            "ex_post_outcome_label": label,
            "label_uses_future_information": True}
