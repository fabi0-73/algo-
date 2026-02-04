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
