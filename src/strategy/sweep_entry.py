"""
Sweep-Reversal Entry — fade a liquidity sweep of a stop-cluster level.

Model (Turtle Soup ancestry, ICT "liquidity raid" framing):
  1. Price pokes >= min_poke_atr_mult * ATR BEYOND a liquidity level
     (running the stops resting there),
  2. within max_candles_back_inside bars price CLOSES back inside the level,
  3. optional rejection-candle confirmation on the trigger bar,
  4. entry fades the sweep: sweep above highs -> SHORT, below lows -> LONG.
     SL beyond the sweep extreme (risk.calculate_risk adds the ATR buffer and
     min-stop floor); TP by exit style (FIXED_RR at tp_rr, or HYBRID
     partial@1R + trail).

Evidence caveat (see plan): this class of edge is thin and cost-sensitive —
the model ships behind SWEEP_MODEL config with explicit kill criteria and is
validated on the honest-cost engine before ever reaching live signals.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from config import SWEEP_MODEL
from src.strategy.liquidity_levels import LiquidityLevel


@dataclass
class SweepSignal:
    """A detected sweep-reversal entry opportunity."""
    valid: bool = False
    direction: str = ""            # "LONG" | "SHORT"
    level_price: float = 0.0
    level_kind: str = ""
    sweep_extreme: float = 0.0     # the wick extreme beyond the level (SL anchor)
    poke_bar_idx: int = -1         # bar that made the extreme (dedupe key)
    entry_idx: int = -1
    poke_atr_mult: float = 0.0     # how far beyond the level, in ATR
    rejection_confirmed: bool = False
    volume_bonus: bool = False


def detect_sweep_at_candle(
    current_idx: int,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    atr: float,
    levels: List[LiquidityLevel],
    cfg: dict = None,
) -> List[SweepSignal]:
    """Check the current bar for completed sweep-reversal triggers.

    Returns every valid signal at this bar (usually 0 or 1). Trigger bar = the
    first bar whose CLOSE is back inside the level after a qualifying poke
    within the window; enforced via 'previous close was still beyond OR the
    poke is this bar' so each sweep fires exactly once.
    """
    cfg = cfg or SWEEP_MODEL
    if not atr or atr <= 0 or current_idx < 1:
        return []

    i = current_idx
    w = int(cfg.get("max_candles_back_inside", 3))
    min_poke = float(cfg.get("min_poke_atr_mult", 0.10)) * atr
    max_dist = float(cfg.get("max_level_distance_atr", 1.5)) * atr
    require_rejection = bool(cfg.get("require_rejection", True))
    lo = max(0, i - w + 1)

    close_i, open_i, high_i, low_i = closes[i], opens[i], highs[i], lows[i]
    signals: List[SweepSignal] = []

    for lvl in levels:
        L = lvl.price
        if abs(close_i - L) > max_dist:
            continue  # trigger close too far from the level to be a fresh fade

        if lvl.side == "ABOVE":
            # Buy-stops above: sweep UP through L, close back BELOW -> SHORT
            if close_i >= L:
                continue
            win_hi = highs[lo:i + 1]
            poke_off = int(np.argmax(win_hi))
            sweep_extreme = float(win_hi[poke_off])
            if sweep_extreme < L + min_poke:
                continue
            # fire once: this bar is the FIRST close back inside
            if closes[i - 1] < L and (lo + poke_off) != i and i - 1 >= lo:
                continue
            rejection = (high_i > L) or (close_i < open_i)  # wick through / bearish body
            if require_rejection and not rejection:
                continue
            signals.append(SweepSignal(
                valid=True, direction="SHORT", level_price=L, level_kind=lvl.kind,
                sweep_extreme=sweep_extreme, poke_bar_idx=lo + poke_off,
                entry_idx=i, poke_atr_mult=(sweep_extreme - L) / atr,
                rejection_confirmed=rejection,
                volume_bonus=_volume_bonus(volumes, lo + poke_off, cfg),
            ))
        else:
            # Sell-stops below: sweep DOWN through L, close back ABOVE -> LONG
            if close_i <= L:
                continue
            win_lo = lows[lo:i + 1]
            poke_off = int(np.argmin(win_lo))
            sweep_extreme = float(win_lo[poke_off])
            if sweep_extreme > L - min_poke:
                continue
            if closes[i - 1] > L and (lo + poke_off) != i and i - 1 >= lo:
                continue
            rejection = (low_i < L) or (close_i > open_i)  # wick through / bullish body
            if require_rejection and not rejection:
                continue
            signals.append(SweepSignal(
                valid=True, direction="LONG", level_price=L, level_kind=lvl.kind,
                sweep_extreme=sweep_extreme, poke_bar_idx=lo + poke_off,
                entry_idx=i, poke_atr_mult=(L - sweep_extreme) / atr,
                rejection_confirmed=rejection,
                volume_bonus=_volume_bonus(volumes, lo + poke_off, cfg),
            ))

    return signals


def _volume_bonus(volumes: np.ndarray, poke_idx: int, cfg: dict) -> bool:
    """Soft confluence: was the sweep bar's tick volume elevated vs recent avg?
    Never a hard gate — MT5 gold 'volume' is tick count, a proxy only."""
    if not cfg.get("volume_bonus", True) or volumes is None or poke_idx < 20:
        return False
    recent = volumes[poke_idx - 20:poke_idx]
    avg = float(np.mean(recent)) if len(recent) else 0.0
    ratio = float(cfg.get("volume_bonus_ratio", 1.5))
    return avg > 0 and float(volumes[poke_idx]) >= ratio * avg
