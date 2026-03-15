"""
Phase 3: Distribution Detection (Real Breakout)
Confirms the real move after the manipulation fakeout.
"""
from dataclasses import dataclass
from typing import Optional
import math
import pandas as pd
import numpy as np

from config import STRATEGY
from .consolidation import ConsolidationResult
from .manipulation import ManipulationResult
from .indicators import calculate_body_sizes, calculate_avg_body_size


@dataclass
class DistributionResult:
    """Result of distribution (real breakout) detection."""
    valid: bool
    direction: str = ""  # "UP" (bullish) or "DOWN" (bearish)
    break_price: float = 0.0  # Close price that confirmed the break
    break_distance: float = 0.0  # How far beyond the boundary
    body_expansion: float = 0.0  # Body size / avg body size ratio
    break_candle_idx: int = 0  # Index of the breakout candle
    atr: float = 0.0


def detect_distribution(
    df: pd.DataFrame,
    consolidation: ConsolidationResult,
    manipulation: ManipulationResult,
    break_atr_mult: float = None,
    body_mult: float = None,
) -> DistributionResult:
    """
    Detect distribution (real breakout) after manipulation.
    
    Distribution criteria:
    1. Price breaks OPPOSITE side of where manipulation occurred
    2. Close beyond boundary by >= break_atr_mult * ATR
    3. Body size > body_mult * average body size
    
    Args:
        df: DataFrame with OHLC data
        consolidation: The consolidation zone
        manipulation: The manipulation (fakeout) result
        break_atr_mult: Break threshold as ATR multiple (default from config)
        body_mult: Body expansion multiplier (default from config)
    
    Returns:
        DistributionResult with detection results
    """
    # Use config defaults
    break_atr_mult = break_atr_mult or STRATEGY["distribution_break_atr_mult"]
    body_mult = body_mult or STRATEGY["distribution_body_mult"]
    
    if not consolidation.valid or not manipulation.valid:
        return DistributionResult(valid=False)
    
    # Start looking after manipulation returns
    start_idx = manipulation.return_candle_idx + 1
    if start_idx >= len(df):
        return DistributionResult(valid=False)
    
    # Look for distribution in next N candles
    search_window = min(20, len(df) - start_idx)
    if search_window < 1:
        return DistributionResult(valid=False)
    
    post_manip = df.iloc[start_idx:start_idx + search_window]
    
    atr = manipulation.atr
    min_break_distance = break_atr_mult * atr
    
    range_high = consolidation.range_high
    range_low = consolidation.range_low
    
    # Calculate average body size from consolidation period
    consol_df = df.iloc[consolidation.start_idx:consolidation.end_idx + 1]
    avg_body = calculate_body_sizes(consol_df).mean()
    
    if avg_body == 0:
        avg_body = atr * 0.1  # Fallback
    
    # Determine expected distribution direction
    # Fakeout UP -> real move DOWN
    # Fakeout DOWN -> real move UP
    expected_direction = "UP" if manipulation.direction == "DOWN" else "DOWN"
    
    # Scan for distribution breakout
    for i, (idx, candle) in enumerate(post_manip.iterrows()):
        body_size = abs(candle["close"] - candle["open"])
        body_ratio = body_size / avg_body if avg_body > 0 else 0
        
        if expected_direction == "UP":
            # Looking for break above range_high
            break_distance = candle["close"] - range_high
            
            if break_distance >= min_break_distance:
                if body_ratio >= body_mult:
                    return DistributionResult(
                        valid=True,
                        direction="UP",
                        break_price=candle["close"],
                        break_distance=break_distance,
                        body_expansion=body_ratio,
                        break_candle_idx=start_idx + i,
                        atr=atr,
                    )
        else:
            # Looking for break below range_low
            break_distance = range_low - candle["close"]
            
            if break_distance >= min_break_distance:
                if body_ratio >= body_mult:
                    return DistributionResult(
                        valid=True,
                        direction="DOWN",
                        break_price=candle["close"],
                        break_distance=break_distance,
                        body_expansion=body_ratio,
                        break_candle_idx=start_idx + i,
                        atr=atr,
                    )
    
    return DistributionResult(valid=False, atr=atr)


def validate_distribution_strength(
    df: pd.DataFrame,
    distribution: DistributionResult,
    min_follow_through_candles: int = 2,
    require_extension: bool = None,
) -> bool:
    """
    Validate that distribution has follow-through (not just a spike).

    Args:
        df: Full DataFrame
        distribution: The distribution result
        min_follow_through_candles: Minimum candles that should continue in direction
        require_extension: If True, follow-through must make new extreme beyond break price

    Returns:
        True if distribution appears strong
    """
    if not distribution.valid:
        return False

    if require_extension is None:
        require_extension = STRATEGY.get("distribution_require_extension", False)

    start_idx = distribution.break_candle_idx + 1
    end_idx = min(start_idx + min_follow_through_candles, len(df))

    if end_idx <= start_idx:
        return True  # Not enough data to validate, assume valid

    follow_through = df.iloc[start_idx:end_idx]

    if distribution.direction == "UP":
        # Check if subsequent candles maintain bullish bias
        bullish_count = (follow_through["close"] > follow_through["open"]).sum()
        if bullish_count < math.ceil(min_follow_through_candles / 2):
            return False
        # Check for new high extension beyond break price
        if require_extension and follow_through["high"].max() <= distribution.break_price:
            return False
        return True
    else:
        # Check if subsequent candles maintain bearish bias
        bearish_count = (follow_through["close"] < follow_through["open"]).sum()
        if bearish_count < math.ceil(min_follow_through_candles / 2):
            return False
        # Check for new low extension beyond break price
        if require_extension and follow_through["low"].min() >= distribution.break_price:
            return False
        return True
