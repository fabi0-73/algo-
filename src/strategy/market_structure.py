"""
Market Structure Detection
Identifies swing points and break of structure (BOS) for trend confirmation.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

from config import STRATEGY


@dataclass
class SwingPoint:
    """Swing high or swing low point."""
    valid: bool
    type: str = ""          # "HIGH" or "LOW"
    price: float = 0.0      # Price level of the swing
    candle_idx: int = 0     # Index of the swing candle
    strength: int = 0       # Number of candles on each side confirming the swing


@dataclass
class StructureBreak:
    """Break of Structure (BOS) representation."""
    valid: bool
    direction: str = ""         # "BULLISH" or "BEARISH"
    broken_level: float = 0.0   # The swing high/low that was broken
    break_price: float = 0.0    # Close price that confirmed the break
    break_candle_idx: int = 0   # Index of the candle that broke structure
    swing_idx: int = 0          # Index of the swing that was broken
    
    @property
    def is_bullish(self) -> bool:
        return self.direction == "BULLISH"
    
    @property
    def is_bearish(self) -> bool:
        return self.direction == "BEARISH"


def detect_swing_high(
    df: pd.DataFrame,
    idx: int,
    lookback: int = 5,
    lookahead: int = 5,
    highs_arr=None,
) -> Optional[SwingPoint]:
    """
    Detect if a candle is a swing high.
    
    A swing high is a candle with a higher high than N candles before and after.
    
    Args:
        df: DataFrame with OHLC data
        idx: Index to check
        lookback: Candles to check before
        lookahead: Candles to check after
        highs_arr: Pre-extracted highs array (avoids df.iloc when provided)
    
    Returns:
        SwingPoint if this is a swing high
    """
    if idx < lookback or idx >= len(df) - lookahead:
        return None
    
    h = highs_arr if highs_arr is not None else df["high"].values
    current_high = h[idx]
    
    for i in range(1, lookback + 1):
        if h[idx - i] >= current_high:
            return None
    
    for i in range(1, lookahead + 1):
        if h[idx + i] >= current_high:
            return None
    
    return SwingPoint(
        valid=True,
        type="HIGH",
        price=float(current_high),
        candle_idx=idx,
        strength=min(lookback, lookahead),
    )


def detect_swing_low(
    df: pd.DataFrame,
    idx: int,
    lookback: int = 5,
    lookahead: int = 5,
    lows_arr=None,
) -> Optional[SwingPoint]:
    """
    Detect if a candle is a swing low.
    
    A swing low is a candle with a lower low than N candles before and after.
    
    Args:
        df: DataFrame with OHLC data
        idx: Index to check
        lookback: Candles to check before
        lookahead: Candles to check after
        lows_arr: Pre-extracted lows array (avoids df.iloc when provided)
    
    Returns:
        SwingPoint if this is a swing low
    """
    if idx < lookback or idx >= len(df) - lookahead:
        return None
    
    l = lows_arr if lows_arr is not None else df["low"].values
    current_low = l[idx]
    
    for i in range(1, lookback + 1):
        if l[idx - i] <= current_low:
            return None
    
    for i in range(1, lookahead + 1):
        if l[idx + i] <= current_low:
            return None
    
    return SwingPoint(
        valid=True,
        type="LOW",
        price=float(current_low),
        candle_idx=idx,
        strength=min(lookback, lookahead),
    )


def find_swing_points(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    swing_lookback: int = None,
    swing_type: str = None,
    highs_arr=None,
    lows_arr=None,
) -> List[SwingPoint]:
    """
    Find all swing points in a range.
    
    Args:
        df: DataFrame with OHLC data
        start_idx: Start of range
        end_idx: End of range
        swing_lookback: Lookback for swing detection
        swing_type: Filter by type ("HIGH", "LOW", or None for both)
        highs_arr: Pre-extracted highs array
        lows_arr: Pre-extracted lows array
    
    Returns:
        List of SwingPoint objects
    """
    swing_lookback = swing_lookback or STRATEGY.get("bos_swing_lookback", 5)
    
    swing_points = []
    
    # Use smaller lookahead for real-time detection
    lookahead = min(swing_lookback, 3)
    
    for idx in range(start_idx + swing_lookback, end_idx - lookahead):
        if swing_type is None or swing_type == "HIGH":
            high = detect_swing_high(df, idx, swing_lookback, lookahead, highs_arr=highs_arr)
            if high:
                swing_points.append(high)
        
        if swing_type is None or swing_type == "LOW":
            low = detect_swing_low(df, idx, swing_lookback, lookahead, lows_arr=lows_arr)
            if low:
                swing_points.append(low)
    
    swing_points.sort(key=lambda x: x.candle_idx)
    
    return swing_points


def find_recent_swing_high(
    df: pd.DataFrame,
    current_idx: int,
    lookback: int = 50,
    swing_strength: int = 3,
    highs_arr=None,
) -> Optional[SwingPoint]:
    """
    Find the most recent swing high before current index.
    
    Args:
        df: DataFrame with OHLC data
        current_idx: Current index
        lookback: How far back to search
        swing_strength: Minimum swing strength
        highs_arr: Pre-extracted highs array
    
    Returns:
        Most recent SwingPoint HIGH or None
    """
    start_idx = max(0, current_idx - lookback)
    
    for idx in range(current_idx - swing_strength - 1, start_idx, -1):
        swing = detect_swing_high(df, idx, swing_strength, swing_strength, highs_arr=highs_arr)
        if swing:
            return swing
    
    return None


def find_recent_swing_low(
    df: pd.DataFrame,
    current_idx: int,
    lookback: int = 50,
    swing_strength: int = 3,
    lows_arr=None,
) -> Optional[SwingPoint]:
    """
    Find the most recent swing low before current index.
    
    Args:
        df: DataFrame with OHLC data
        current_idx: Current index
        lookback: How far back to search
        swing_strength: Minimum swing strength
        lows_arr: Pre-extracted lows array
    
    Returns:
        Most recent SwingPoint LOW or None
    """
    start_idx = max(0, current_idx - lookback)
    
    for idx in range(current_idx - swing_strength - 1, start_idx, -1):
        swing = detect_swing_low(df, idx, swing_strength, swing_strength, lows_arr=lows_arr)
        if swing:
            return swing
    
    return None


def detect_break_of_structure(
    df: pd.DataFrame,
    candle_idx: int,
    swing: SwingPoint,
    require_close: bool = True,
    closes_arr=None,
    highs_arr=None,
    lows_arr=None,
) -> Optional[StructureBreak]:
    """
    Detect if a candle breaks a swing point (Break of Structure).
    
    Args:
        df: DataFrame with OHLC data
        candle_idx: Index of candle to check
        swing: The swing point to check against
        require_close: Whether to require close beyond swing (not just wick)
        closes_arr: Pre-extracted closes array
        highs_arr: Pre-extracted highs array
        lows_arr: Pre-extracted lows array
    
    Returns:
        StructureBreak if BOS occurred
    """
    if candle_idx >= len(df) or candle_idx <= swing.candle_idx:
        return None
    
    c = closes_arr if closes_arr is not None else df["close"].values
    h = highs_arr if highs_arr is not None else df["high"].values
    l_ = lows_arr if lows_arr is not None else df["low"].values

    if swing.type == "HIGH":
        if require_close:
            breaks = c[candle_idx] > swing.price
        else:
            breaks = h[candle_idx] > swing.price
        
        if breaks:
            return StructureBreak(
                valid=True,
                direction="BULLISH",
                broken_level=swing.price,
                break_price=float(c[candle_idx]),
                break_candle_idx=candle_idx,
                swing_idx=swing.candle_idx,
            )
    else:
        if require_close:
            breaks = c[candle_idx] < swing.price
        else:
            breaks = l_[candle_idx] < swing.price
        
        if breaks:
            return StructureBreak(
                valid=True,
                direction="BEARISH",
                broken_level=swing.price,
                break_price=float(c[candle_idx]),
                break_candle_idx=candle_idx,
                swing_idx=swing.candle_idx,
            )
    
    return None


def find_bos_after_manipulation(
    df: pd.DataFrame,
    manipulation_return_idx: int,
    expected_direction: str,
    search_window: int = 20,
    swing_lookback: int = None,
    highs_arr=None,
    lows_arr=None,
    closes_arr=None,
    current_idx: int = None,
) -> Optional[StructureBreak]:
    """
    Find Break of Structure after manipulation phase.
    
    For bullish setup (manipulation DOWN):
    - Find swing high in the consolidation/manipulation zone
    - Check if price breaks above it
    
    For bearish setup (manipulation UP):
    - Find swing low in the consolidation/manipulation zone
    - Check if price breaks below it
    
    Args:
        df: DataFrame with OHLC data
        manipulation_return_idx: Index where manipulation completed
        expected_direction: Expected BOS direction ("BULLISH" or "BEARISH")
        search_window: Candles to search after manipulation
        swing_lookback: Swing detection lookback
        highs_arr: Pre-extracted highs array
        lows_arr: Pre-extracted lows array
        closes_arr: Pre-extracted closes array
        current_idx: Last bar knowable at decision time. Bars beyond it are
            never scanned (backtest causality; live frames simply end here).

    Returns:
        StructureBreak if found
    """
    swing_lookback = swing_lookback or STRATEGY.get("bos_swing_lookback", 5)
    
    if expected_direction == "BULLISH":
        swing = find_recent_swing_high(
            df,
            manipulation_return_idx,
            lookback=30,
            swing_strength=min(swing_lookback, 3),
            highs_arr=highs_arr,
        )
    else:
        swing = find_recent_swing_low(
            df,
            manipulation_return_idx,
            lookback=30,
            swing_strength=min(swing_lookback, 3),
            lows_arr=lows_arr,
        )
    
    if swing is None:
        return None
    
    end_idx = min(manipulation_return_idx + search_window, len(df))
    if current_idx is not None:
        # Decision at current_idx may only see completed bars <= current_idx
        end_idx = min(end_idx, current_idx + 1)

    for idx in range(manipulation_return_idx + 1, end_idx):
        bos = detect_break_of_structure(
            df, idx, swing, require_close=True,
            closes_arr=closes_arr, highs_arr=highs_arr, lows_arr=lows_arr,
        )
        if bos:
            return bos
    
    return None


def validate_structure_break(
    df: pd.DataFrame,
    bos: StructureBreak,
    min_displacement_pips: float = 0,
    closes_arr=None,
) -> bool:
    """
    Validate that a BOS has follow-through (not just a wick spike).
    
    Args:
        df: DataFrame with OHLC data
        bos: The StructureBreak to validate
        min_displacement_pips: Minimum move beyond broken level
        closes_arr: Pre-extracted closes array
    
    Returns:
        True if BOS appears valid
    """
    if not bos.valid:
        return False
    
    next_idx = bos.break_candle_idx + 1
    if next_idx >= len(df):
        return True  # Can't validate, assume valid
    
    c = closes_arr if closes_arr is not None else df["close"].values
    
    if bos.direction == "BULLISH":
        return c[next_idx] > bos.broken_level
    else:
        return c[next_idx] < bos.broken_level


def get_market_structure_bias(
    df: pd.DataFrame,
    current_idx: int,
    lookback: int = 50,
) -> str:
    """
    Determine overall market structure bias.
    
    Args:
        df: DataFrame with OHLC data
        current_idx: Current index
        lookback: How far back to analyze
    
    Returns:
        "BULLISH", "BEARISH", or "NEUTRAL"
    """
    start_idx = max(0, current_idx - lookback)
    
    # Find swing points
    swings = find_swing_points(df, start_idx, current_idx)
    
    if len(swings) < 4:
        return "NEUTRAL"
    
    # Analyze structure: higher highs + higher lows = bullish
    # Lower highs + lower lows = bearish
    highs = [s for s in swings if s.type == "HIGH"]
    lows = [s for s in swings if s.type == "LOW"]
    
    if len(highs) < 2 or len(lows) < 2:
        return "NEUTRAL"
    
    # Check last two of each
    higher_highs = highs[-1].price > highs[-2].price
    higher_lows = lows[-1].price > lows[-2].price
    lower_highs = highs[-1].price < highs[-2].price
    lower_lows = lows[-1].price < lows[-2].price
    
    if higher_highs and higher_lows:
        return "BULLISH"
    elif lower_highs and lower_lows:
        return "BEARISH"
    else:
        return "NEUTRAL"
