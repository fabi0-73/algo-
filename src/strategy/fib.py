"""Fibonacci retracement helpers for entry-price refinement (OTE zone).

Evidence note (2026-07-04 research pass): academic results on Fibonacci levels are
null — retracement depths are continuously distributed with no privileged ratios
(Expert Systems w/ Applications 2021; arXiv:1605.03559). What IS real is the
geometry: entering deeper into a retracement shrinks the stop distance to the
protective extreme and widens the realized R multiple on the same setups. The
0.618-0.79 band is therefore used purely as a "deep pullback" definition, not as
a claim that those ratios carry information.
"""

from typing import Tuple


def fib_zone(
    leg_start: float,
    leg_end: float,
    low_pct: float = 0.618,
    high_pct: float = 0.79,
) -> Tuple[float, float]:
    """
    Price band for a retracement of `low_pct`..`high_pct` of the leg from
    leg_start -> leg_end, measured back from leg_end toward leg_start.

    For a bullish leg (start < end) the zone sits below leg_end; for a bearish
    leg above it. Returns (zone_lo, zone_hi) with zone_lo <= zone_hi.
    """
    a = leg_end - (leg_end - leg_start) * low_pct
    b = leg_end - (leg_end - leg_start) * high_pct
    return (min(a, b), max(a, b))


def is_in_ote(
    price: float,
    leg_start: float,
    leg_end: float,
    low_pct: float = 0.618,
    high_pct: float = 0.79,
) -> bool:
    """True if price lies inside the OTE (deep-pullback) band of the leg."""
    lo, hi = fib_zone(leg_start, leg_end, low_pct, high_pct)
    return lo <= price <= hi


def ote_entry_price(
    direction: str,
    leg_start: float,
    leg_end: float,
    retest_level: float,
    low_pct: float = 0.618,
    high_pct: float = 0.79,
) -> float:
    """
    Entry price refined toward the OTE band, never worse than the retest level.

    LONG: the shallow OTE edge (closer to leg_end) capped at retest_level from
    below — we ask for a fill at min(retest_level, shallow_edge) so the limit
    only ever improves on the plain retest price. SHORT mirrors upward.
    """
    lo, hi = fib_zone(leg_start, leg_end, low_pct, high_pct)
    if direction == "LONG":
        return min(retest_level, hi)
    return max(retest_level, lo)
