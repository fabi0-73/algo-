"""
Phase 1: Consolidation Detection
Identifies tight range consolidation zones where liquidity accumulates.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from config import STRATEGY
from .indicators import calculate_atr, calculate_range_boundaries


@dataclass
class ConsolidationResult:
    """Result of consolidation detection."""
    valid: bool
    range_high: float = 0.0
    range_low: float = 0.0
    range_size: float = 0.0
    atr: float = 0.0
    close_inside_pct: float = 0.0
    start_idx: int = 0
    end_idx: int = 0
    # Equal highs/lows (liquidity pools) within consolidation
    has_equal_highs: bool = False
    has_equal_lows: bool = False
    equal_high_level: float = 0.0
    equal_low_level: float = 0.0

    @property
    def range_mid(self) -> float:
        """Middle of the consolidation range."""
        return (self.range_high + self.range_low) / 2


def detect_consolidation(
    df: pd.DataFrame,
    lookback: int = None,
    range_atr_mult: float = None,
    close_pct_threshold: float = None,
) -> ConsolidationResult:
    """
    Detect if the last N candles form a consolidation zone.
    
    Consolidation criteria:
    1. Range (high-low) must be <= range_atr_mult * ATR(14)
    2. At least close_pct_threshold of closes must be inside range
    
    Args:
        df: DataFrame with OHLC data (must have enough history for ATR)
        lookback: Number of candles to analyze (default from config)
        range_atr_mult: Range threshold as multiple of ATR (default from config)
        close_pct_threshold: Minimum percentage of closes inside range (default from config)
    
    Returns:
        ConsolidationResult with detection results
    """
    # Use config defaults
    lookback = lookback or STRATEGY["consolidation_lookback"]
    range_atr_mult = range_atr_mult or STRATEGY["consolidation_range_atr_mult"]
    close_pct_threshold = close_pct_threshold or STRATEGY["consolidation_close_pct"]
    
    # Need enough data
    if len(df) < lookback + STRATEGY["atr_period"]:
        return ConsolidationResult(valid=False)
    
    # Calculate ATR on full data, then get last value
    atr_series = calculate_atr(df, period=STRATEGY["atr_period"])
    current_atr = atr_series.iloc[-1]
    
    if pd.isna(current_atr) or current_atr == 0:
        return ConsolidationResult(valid=False)
    
    # Get the consolidation window
    window = df.iloc[-lookback:]
    
    # Calculate range boundaries
    range_high, range_low = calculate_range_boundaries(window)
    range_size = range_high - range_low
    
    # Check range condition: range <= 0.35 * ATR
    max_allowed_range = range_atr_mult * current_atr
    if range_size > max_allowed_range:
        return ConsolidationResult(
            valid=False,
            range_high=range_high,
            range_low=range_low,
            range_size=range_size,
            atr=current_atr,
        )
    
    # Check closes inside range
    # A close is "inside" if it's between range_low and range_high
    closes = window["close"]
    closes_inside = ((closes >= range_low) & (closes <= range_high)).sum()
    close_inside_pct = closes_inside / len(window)
    
    # Check close percentage condition: >= 70% inside
    if close_inside_pct < close_pct_threshold:
        return ConsolidationResult(
            valid=False,
            range_high=range_high,
            range_low=range_low,
            range_size=range_size,
            atr=current_atr,
            close_inside_pct=close_inside_pct,
        )
    
    # Valid consolidation found
    return ConsolidationResult(
        valid=True,
        range_high=range_high,
        range_low=range_low,
        range_size=range_size,
        atr=current_atr,
        close_inside_pct=close_inside_pct,
        start_idx=len(df) - lookback,
        end_idx=len(df) - 1,
    )


def score_consolidation_quality(consol: ConsolidationResult) -> int:
    """
    Score consolidation quality (0-5). Higher = better setup.

    Scoring:
    +1 if range_size / atr < 1.5 (tight range)
    +1 if close_inside_pct >= 0.80 (clean consolidation)
    +1 if has equal highs or equal lows (liquidity pool present)
    +1 if has BOTH equal highs AND equal lows (double liquidity)
    +1 if duration >= 10 bars (sustained accumulation)
    """
    if not consol.valid or consol.atr <= 0:
        return 0

    score = 0

    # Tight range relative to ATR
    if consol.range_size / consol.atr < 1.5:
        score += 1

    # High percentage of closes inside range
    if consol.close_inside_pct >= 0.80:
        score += 1

    # Has at least one equal level (liquidity pool)
    if consol.has_equal_highs or consol.has_equal_lows:
        score += 1

    # Has both equal highs AND lows (double liquidity)
    if consol.has_equal_highs and consol.has_equal_lows:
        score += 1

    # Sustained consolidation (longer duration)
    duration = consol.end_idx - consol.start_idx
    if duration >= 10:
        score += 1

    return score


def find_consolidation_zones(
    df: pd.DataFrame,
    lookback: int = None,
    range_atr_mult: float = None,
    close_pct_threshold: float = None,
    min_gap: int = 5,
) -> list:
    """
    Scan through data to find all consolidation zones.
    
    Args:
        df: Full DataFrame with OHLC data
        lookback: Consolidation lookback period
        range_atr_mult: Range threshold multiplier
        close_pct_threshold: Minimum close percentage inside
        min_gap: Minimum candles between consolidation zones
    
    Returns:
        List of ConsolidationResult objects
    """
    lookback = lookback or STRATEGY["consolidation_lookback"]
    zones = []
    last_zone_end = 0
    
    for i in range(lookback + STRATEGY["atr_period"], len(df)):
        # Skip if too close to last zone
        if i < last_zone_end + min_gap:
            continue
        
        # Test this position
        test_df = df.iloc[:i+1]
        result = detect_consolidation(
            test_df,
            lookback=lookback,
            range_atr_mult=range_atr_mult,
            close_pct_threshold=close_pct_threshold,
        )
        
        if result.valid:
            zones.append(result)
            last_zone_end = result.end_idx
    
    return zones


def detect_equal_levels(
    df: pd.DataFrame,
    consolidation: ConsolidationResult,
    tolerance_atr_mult: float = None,
    min_touches: int = None,
    highs_arr: np.ndarray = None,
    lows_arr: np.ndarray = None,
) -> ConsolidationResult:
    """
    Detect equal highs/lows within consolidation (liquidity pools).

    Equal highs/lows are where multiple candles touch the same level,
    indicating obvious stop placement by retail traders.

    Args:
        df: Full DataFrame with OHLC data
        consolidation: The consolidation result to enrich
        tolerance_atr_mult: How close highs/lows must be to count as equal (default from config)
        min_touches: Minimum touches to confirm equal level (default from config)
        highs_arr: Pre-extracted highs numpy array (avoids df slicing when provided)
        lows_arr: Pre-extracted lows numpy array (avoids df slicing when provided)

    Returns:
        ConsolidationResult with equal level fields set
    """
    if not consolidation.valid:
        return consolidation

    tolerance_atr_mult = tolerance_atr_mult or STRATEGY.get("equal_level_tolerance_atr_mult", 0.05)
    min_touches = min_touches or STRATEGY.get("equal_level_min_touches", 2)

    if not STRATEGY.get("detect_equal_levels", True):
        return consolidation

    start = consolidation.start_idx
    end = consolidation.end_idx + 1
    if start >= len(df) or end > len(df):
        return consolidation

    tolerance = tolerance_atr_mult * consolidation.atr

    if highs_arr is not None:
        highs = highs_arr[start:end]
    else:
        highs = df.iloc[start:end]["high"].values

    for level in highs:
        touches = sum(1 for h in highs if abs(h - level) <= tolerance)
        if touches >= min_touches:
            consolidation.has_equal_highs = True
            consolidation.equal_high_level = float(level)
            break

    if lows_arr is not None:
        lows = lows_arr[start:end]
    else:
        lows = df.iloc[start:end]["low"].values

    for level in lows:
        touches = sum(1 for l in lows if abs(l - level) <= tolerance)
        if touches >= min_touches:
            consolidation.has_equal_lows = True
            consolidation.equal_low_level = float(level)
            break

    return consolidation
