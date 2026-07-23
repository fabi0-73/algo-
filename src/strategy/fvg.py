"""
Fair Value Gap (FVG) Detection
Identifies price imbalances where institutional activity creates gaps.
"""
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np

from config import STRATEGY


@dataclass
class FVG:
    """Fair Value Gap representation."""
    valid: bool
    direction: str = ""      # "BULLISH" or "BEARISH"
    top: float = 0.0         # Upper boundary of FVG
    bottom: float = 0.0      # Lower boundary of FVG
    candle_idx: int = 0      # Index where FVG formed (middle candle)
    filled: bool = False     # Whether FVG has been filled
    fill_idx: int = 0        # Index where FVG was filled
    # Quality scoring fields
    quality_score: int = 0           # Quality score (0-3)
    impulse_body_ratio: float = 0.0  # Body ratio of impulse candle
    
    @property
    def midpoint(self) -> float:
        """Middle of the FVG zone."""
        return (self.top + self.bottom) / 2
    
    @property
    def size(self) -> float:
        """Size of the FVG."""
        return self.top - self.bottom
    
    def contains_price(self, price: float) -> bool:
        """Check if a price is within the FVG zone."""
        return self.bottom <= price <= self.top
    
    def is_near(self, level: float, tolerance: float) -> bool:
        """Check if FVG is near a given price level."""
        return (self.bottom - tolerance <= level <= self.top + tolerance)


def detect_fvg(
    df: pd.DataFrame,
    idx: int,
    min_size_atr_mult: float = None,
    atr: float = None,
) -> Optional[FVG]:
    """
    Detect if a Fair Value Gap exists at a specific index.
    
    FVG Definition:
    - Bullish FVG: Gap between candle[i-2].high and candle[i].low
      (candle i-1 is the impulse candle that created the gap)
    - Bearish FVG: Gap between candle[i-2].low and candle[i].high
    
    Args:
        df: DataFrame with OHLC data
        idx: Index to check (this is the 3rd candle of the pattern)
        min_size_atr_mult: Minimum FVG size as ATR multiple
        atr: ATR value for size validation
    
    Returns:
        FVG object if found, None otherwise
    """
    min_size_atr_mult = min_size_atr_mult or STRATEGY.get("fvg_min_size_atr_mult", 0.10)
    return _detect_fvg_core(df["high"].values, df["low"].values, len(df),
                            idx, min_size_atr_mult, atr)


def _detect_fvg_core(highs, lows, n, idx, min_size_atr_mult, atr):
    """Array-core of detect_fvg — identical logic on plain numpy columns
    (the pandas per-row .iloc access dominated the whole backtest profile)."""
    # Need at least 3 candles
    if idx < 2 or idx >= n:
        return None

    c1_high = highs[idx - 2]
    c1_low = lows[idx - 2]
    c3_low = lows[idx]
    c3_high = highs[idx]

    # Check for Bullish FVG (gap up)
    # Gap exists when candle 1's high is below candle 3's low
    if c1_high < c3_low:
        fvg_bottom = c1_high
        fvg_top = c3_low
        fvg_size = fvg_top - fvg_bottom

        # Validate minimum size
        if atr is not None and min_size_atr_mult > 0:
            if fvg_size < min_size_atr_mult * atr:
                return None

        return FVG(
            valid=True,
            direction="BULLISH",
            top=fvg_top,
            bottom=fvg_bottom,
            candle_idx=idx - 1,  # Middle candle created the FVG
        )

    # Check for Bearish FVG (gap down)
    # Gap exists when candle 1's low is above candle 3's high
    if c1_low > c3_high:
        fvg_top = c1_low
        fvg_bottom = c3_high
        fvg_size = fvg_top - fvg_bottom

        # Validate minimum size
        if atr is not None and min_size_atr_mult > 0:
            if fvg_size < min_size_atr_mult * atr:
                return None

        return FVG(
            valid=True,
            direction="BEARISH",
            top=fvg_top,
            bottom=fvg_bottom,
            candle_idx=idx - 1,
        )

    return None


def score_fvg(
    fvg: FVG,
    df: pd.DataFrame,
    atr: float,
) -> FVG:
    """
    Calculate quality score for an FVG.

    Scoring:
    - +1 if size > 0.3 * ATR (large gap)
    - +1 if size > 0.5 * ATR (very large gap, cumulative)
    - +1 if impulse candle body ratio > 0.7 (strong conviction)

    Args:
        fvg: FVG to score
        df: DataFrame with OHLC data
        atr: ATR value

    Returns:
        FVG with quality_score populated
    """
    return _score_fvg_core(df["open"].values, df["high"].values,
                           df["low"].values, df["close"].values, len(df),
                           fvg, atr)


def _score_fvg_core(opens, highs, lows, closes, n, fvg, atr):
    """Array-core of score_fvg — identical logic on plain numpy columns."""
    if not fvg.valid or atr <= 0:
        return fvg

    score = 0
    large_mult = STRATEGY.get("fvg_quality_large_mult", 0.3)
    xlarge_mult = STRATEGY.get("fvg_quality_xlarge_mult", 0.5)
    body_ratio_threshold = STRATEGY.get("fvg_impulse_body_ratio", 0.7)

    # Size scoring
    if fvg.size > large_mult * atr:
        score += 1
    if fvg.size > xlarge_mult * atr:
        score += 1

    # Impulse candle body ratio scoring
    if fvg.candle_idx < n:
        i = fvg.candle_idx
        body = abs(closes[i] - opens[i])
        total_range = highs[i] - lows[i]

        if total_range > 0:
            body_ratio = body / total_range
            fvg.impulse_body_ratio = body_ratio
            if body_ratio > body_ratio_threshold:
                score += 1

    fvg.quality_score = score
    return fvg


def get_fvg_entry_level(fvg: FVG, style: str = None) -> float:
    """
    Get entry level for FVG based on entry style.

    Args:
        fvg: The FVG to get entry level for
        style: Entry style:
            - "equilibrium": 50% of FVG (midpoint) - default
            - "optimal": Best price (furthest into FVG)
            - "exit": Edge of FVG (most conservative)

    Returns:
        Entry price level
    """
    style = style or STRATEGY.get("fvg_entry_style", "equilibrium")

    if not fvg.valid:
        return 0.0

    if style == "equilibrium":
        # 50% of FVG (midpoint)
        return fvg.midpoint
    elif style == "optimal":
        # Best price entry (furthest into FVG)
        # For bullish FVG: bottom is best (lowest entry for long)
        # For bearish FVG: top is best (highest entry for short)
        if fvg.direction == "BULLISH":
            return fvg.bottom
        else:
            return fvg.top
    elif style == "exit":
        # Edge of FVG (most conservative)
        if fvg.direction == "BULLISH":
            return fvg.top
        else:
            return fvg.bottom
    else:
        return fvg.midpoint


def find_fvgs_in_range(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    direction: str = None,
    atr: float = None,
    min_size_atr_mult: float = None,
    min_quality_score: int = None,
) -> List[FVG]:
    """
    Find all FVGs in a given range of candles with optional quality filtering.

    Args:
        df: DataFrame with OHLC data
        start_idx: Starting index (inclusive)
        end_idx: Ending index (inclusive)
        direction: Filter by direction ("BULLISH" or "BEARISH"), None for both
        atr: ATR value for size validation
        min_size_atr_mult: Minimum FVG size as ATR multiple
        min_quality_score: Minimum quality score (0-3) to include FVG

    Returns:
        List of FVG objects found (scored if atr provided)
    """
    fvgs = []
    min_quality = min_quality_score if min_quality_score is not None else STRATEGY.get("fvg_min_quality_score", 0)
    msm = min_size_atr_mult or STRATEGY.get("fvg_min_size_atr_mult", 0.10)

    # Extract columns ONCE — per-row .iloc in this loop was the single
    # hottest path of the whole backtest (see 2026-07-23 profile).
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    # Need at least 3 candles to form FVG
    search_start = max(start_idx, 2)
    search_end = min(end_idx + 1, n)

    for idx in range(search_start, search_end):
        fvg = _detect_fvg_core(highs, lows, n, idx, msm, atr)

        if fvg is not None:
            if direction is None or fvg.direction == direction:
                # Score the FVG if ATR is available
                if atr is not None and atr > 0:
                    fvg = _score_fvg_core(opens, highs, lows, closes, n,
                                          fvg, atr)

                # Filter by quality score
                if fvg.quality_score >= min_quality:
                    fvgs.append(fvg)

    return fvgs


def check_fvg_fill(
    fvg: FVG,
    df: pd.DataFrame,
    start_idx: int = None,
    end_idx: int = None,
) -> FVG:
    """
    Check if an FVG has been filled (price returned to it).
    
    Args:
        fvg: The FVG to check
        df: DataFrame with OHLC data
        start_idx: Start checking from this index (default: after FVG formed)
        end_idx: Stop checking at this index (default: end of data)
    
    Returns:
        Updated FVG with fill status
    """
    if not fvg.valid:
        return fvg
    
    start_idx = start_idx or (fvg.candle_idx + 2)
    end_idx = end_idx or len(df)
    
    for idx in range(start_idx, end_idx):
        candle = df.iloc[idx]
        
        if fvg.direction == "BULLISH":
            # FVG filled when price trades back down into the gap
            if candle["low"] <= fvg.top:
                fvg.filled = True
                fvg.fill_idx = idx
                return fvg
        else:
            # FVG filled when price trades back up into the gap
            if candle["high"] >= fvg.bottom:
                fvg.filled = True
                fvg.fill_idx = idx
                return fvg
    
    return fvg


def find_nearest_fvg(
    fvgs: List[FVG],
    price_level: float,
    direction: str = None,
    max_distance: float = None,
) -> Optional[FVG]:
    """
    Find the FVG nearest to a given price level.
    
    Args:
        fvgs: List of FVGs to search
        price_level: Target price level
        direction: Filter by direction
        max_distance: Maximum distance from price level
    
    Returns:
        Nearest FVG or None if not found
    """
    if not fvgs:
        return None
    
    candidates = fvgs
    if direction:
        candidates = [f for f in fvgs if f.direction == direction]
    
    if not candidates:
        return None
    
    # Sort by distance to price level
    def distance_to_level(fvg: FVG) -> float:
        if fvg.contains_price(price_level):
            return 0
        return min(abs(fvg.top - price_level), abs(fvg.bottom - price_level))
    
    candidates.sort(key=distance_to_level)
    nearest = candidates[0]
    
    # Check max distance
    if max_distance is not None:
        if distance_to_level(nearest) > max_distance:
            return None
    
    return nearest


def find_fvg_at_retest_level(
    df: pd.DataFrame,
    retest_level: float,
    search_start_idx: int,
    search_end_idx: int,
    direction: str,
    atr: float,
    tolerance_mult: float = 0.5,
) -> Optional[FVG]:
    """
    Find an FVG near the retest level that could be used for entry.
    
    Args:
        df: DataFrame with OHLC data
        retest_level: The price level being retested
        search_start_idx: Start of search range
        search_end_idx: End of search range
        direction: Expected FVG direction ("BULLISH" or "BEARISH")
        atr: ATR value
        tolerance_mult: How close FVG must be to retest level (ATR multiple)
    
    Returns:
        FVG if found near retest level
    """
    fvgs = find_fvgs_in_range(
        df,
        start_idx=search_start_idx,
        end_idx=search_end_idx,
        direction=direction,
        atr=atr,
    )
    
    if not fvgs:
        return None
    
    tolerance = tolerance_mult * atr
    
    # Find FVG that overlaps or is very close to retest level
    for fvg in fvgs:
        if fvg.is_near(retest_level, tolerance):
            return fvg
    
    return None


def is_price_leaving_fvg(
    candle: pd.Series,
    fvg: FVG,
) -> bool:
    """
    Check if price is leaving an FVG (entry trigger).
    
    For bullish FVG: Price dipped into FVG and is now closing above it
    For bearish FVG: Price spiked into FVG and is now closing below it
    
    Args:
        candle: Current candle data
        fvg: The FVG to check
    
    Returns:
        True if price is leaving the FVG in the expected direction
    """
    if not fvg.valid:
        return False
    
    if fvg.direction == "BULLISH":
        # Price tapped into FVG (low entered the zone) and is closing above
        entered_fvg = candle["low"] <= fvg.top
        closing_above = candle["close"] > fvg.top
        return entered_fvg and closing_above
    else:
        # Price spiked into FVG (high entered the zone) and is closing below
        entered_fvg = candle["high"] >= fvg.bottom
        closing_below = candle["close"] < fvg.bottom
        return entered_fvg and closing_below
