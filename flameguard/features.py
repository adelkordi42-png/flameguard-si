"""Published-threshold margins for a scenario.

The decision models classify text; they do not subtract O2 percentages or
compare flow to a blow-off limit. This module does that arithmetic once, from
the published values in `data/LABEL_POLICY.md`, and exposes it as four numeric
features the calibrated combiner can use even when no engine is reachable.
"""

from __future__ import annotations

from typing import Any

LIMIT_O2 = {
    "PMMA": 17.0,
    "SIBAL_fabric": 17.0,
    "silicone": 18.0,
    "pmma_nomex": 17.0,
    "methanol": 16.0,
    "heptane": 15.0,
}
BLOWOFF_FLOW = {
    "PMMA": 45.0,
    "SIBAL_fabric": 45.0,
    "silicone": 25.0,
    "pmma_nomex": 25.0,
}
THIN_MM, MEDIUM_MM = 1.0, 3.0
FEATURE_ORDER = ("o2_margin", "flow_fraction", "near_limit", "thickness_regime")


def margins(values: dict[str, Any]) -> dict[str, float]:
    """The four combiner features plus the human-readable limits behind them."""
    material = str(values.get("material") or "")
    o2 = float(values.get("o2_percent") or 0.0)
    flow = float(values.get("opposed_flow_cm_s") or 0.0)
    thickness = float(values.get("thickness_mm") or 0.0)

    o2_limit = LIMIT_O2.get(material, 17.0)
    o2_margin = o2 - o2_limit
    blowoff = BLOWOFF_FLOW.get(material)
    flow_fraction = (flow / blowoff) if blowoff else 0.0
    near_limit = abs(o2_margin) <= 1.0 or (blowoff is not None and 0.8 <= flow_fraction <= 1.2)
    if thickness <= THIN_MM:
        regime = 0.0
    elif thickness <= MEDIUM_MM:
        regime = 1.0
    else:
        regime = 2.0
    return {
        "o2_limit_percent": round(o2_limit, 2),
        "o2_margin_percent": round(o2_margin, 3),
        "blowoff_flow_cm_s": round(blowoff, 2) if blowoff is not None else 0.0,
        "flow_fraction": round(flow_fraction, 4),
        "near_limit": 1.0 if near_limit else 0.0,
        "thickness_regime": regime,
    }


def vector(values: dict[str, Any]) -> list[float]:
    """The four features in `combiner.FEATURE_ORDER` order."""
    computed = margins(values)
    return [
        computed["o2_margin_percent"],
        computed["flow_fraction"],
        computed["near_limit"],
        computed["thickness_regime"],
    ]


def signals(values: dict[str, Any]) -> dict[str, float]:
    """Threshold-based stand-ins for the engine signals, used by reason codes."""
    computed = margins(values)
    o2_margin = computed["o2_margin_percent"]
    flow_fraction = computed["flow_fraction"]
    chemistry = 1.0 if o2_margin > 1.0 else 0.0 if o2_margin < -1.0 else 0.5
    flow = 1.0 if flow_fraction > 1.2 else 0.0 if flow_fraction < 0.8 else 0.5
    return {
        "o2_above_limiting": chemistry,
        "flow_above_blowoff": flow,
        "self_sustaining_60s": chemistry,
        "spread_beyond_ignition": chemistry,
        "severity": 2.5 if chemistry == 1.0 else 1.0 if chemistry == 0.0 else 1.75,
    }