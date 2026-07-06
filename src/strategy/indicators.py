"""
Technical Indicators
ATR, body size, and other calculations needed for AMD strategy.
"""
import numpy as np
import pandas as pd
from typing import Union


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame with OHLC data
        period: ATR period (default 14)
        high_col: Column name for high prices
        low_col: Column name for low prices
        close_col: Column name for close prices
    
    Returns:
        Series with ATR values
    """
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    # True Range is max of all three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR is exponential moving average of TR
    atr = tr.ewm(span=period, adjust=False).mean()
    
    return atr


def calculate_body_sizes(
    df: pd.DataFrame,
    open_col: str = "open",
    close_col: str = "close",
) -> pd.Series:
    """
    Calculate absolute body size of candles.
    
    Args:
        df: DataFrame with OHLC data
        open_col: Column name for open prices
        close_col: Column name for close prices
    
    Returns:
        Series with body sizes
    """
    return abs(df[close_col] - df[open_col])


def calculate_avg_body_size(
    df: pd.DataFrame,
    period: int = 20,
    open_col: str = "open",
    close_col: str = "close",
) -> pd.Series:
    """
    Calculate rolling average body size.
    
    Args:
        df: DataFrame with OHLC data
        period: Lookback period
        open_col: Column name for open prices
        close_col: Column name for close prices
    
    Returns:
        Series with rolling average body sizes
    """
    body_sizes = calculate_body_sizes(df, open_col, close_col)
    return body_sizes.rolling(window=period).mean()


def calculate_range(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
) -> float:
    """
    Calculate the range (highest high - lowest low) of a DataFrame.
    
    Args:
        df: DataFrame with OHLC data
        high_col: Column name for high prices
        low_col: Column name for low prices
    
    Returns:
        Range value
    """
    return df[high_col].max() - df[low_col].min()


def calculate_range_boundaries(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
) -> tuple:
    """
    Calculate the range boundaries (highest high, lowest low).
    
    Args:
        df: DataFrame with OHLC data
        high_col: Column name for high prices
        low_col: Column name for low prices
    
    Returns:
        Tuple of (range_high, range_low)
    """
    return df[high_col].max(), df[low_col].min()


def is_bullish_candle(candle: pd.Series) -> bool:
    """Check if a candle is bullish (close > open)."""
    return candle["close"] > candle["open"]


def is_bearish_candle(candle: pd.Series) -> bool:
    """Check if a candle is bearish (close < open)."""
    return candle["close"] < candle["open"]


def calculate_upper_wick(candle: pd.Series) -> float:
    """Calculate upper wick size of a candle."""
    body_top = max(candle["open"], candle["close"])
    return candle["high"] - body_top


def calculate_lower_wick(candle: pd.Series) -> float:
    """Calculate lower wick size of a candle."""
    body_bottom = min(candle["open"], candle["close"])
    return body_bottom - candle["low"]


def calculate_body_size(candle: pd.Series) -> float:
    """Calculate body size of a single candle."""
    return abs(candle["close"] - candle["open"])


def is_rejection_candle(
    candle: pd.Series,
    direction: str,
    wick_ratio: float = 0.5,
) -> bool:
    """
    Check if candle shows rejection (long wick on one side).
    
    Args:
        candle: Single candle row
        direction: Expected rejection direction ("UP" for bullish rejection, "DOWN" for bearish)
        wick_ratio: Minimum wick to body ratio (default 0.5 = 50%)
    
    Returns:
        True if rejection pattern detected
    """
    body = calculate_body_size(candle)

    if body == 0:
        # Doji: require a GENUINE, dominant rejection wick — not merely a hair's
        # difference. (Bug fix: previously any doji with rejection_wick > opposite_wick
        # by even one tick counted as a rejection, so flat / near-symmetric dojis on
        # M5 gold were wrongly treated as valid retest rejections.)
        lower_wick = calculate_lower_wick(candle)
        upper_wick = calculate_upper_wick(candle)
        rej_wick, opp_wick = (lower_wick, upper_wick) if direction == "UP" else (upper_wick, lower_wick)
        return rej_wick > 0 and rej_wick >= 2.0 * opp_wick

    if direction == "UP":
        # Bullish rejection - long lower wick
        lower_wick = calculate_lower_wick(candle)
        return lower_wick >= body * wick_ratio
    else:
        # Bearish rejection - long upper wick
        upper_wick = calculate_upper_wick(candle)
        return upper_wick >= body * wick_ratio


def add_indicators(
    df: pd.DataFrame,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Add all indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLC data
        atr_period: ATR calculation period
    
    Returns:
        DataFrame with additional indicator columns
    """
    df = df.copy()
    df["atr"] = calculate_atr(df, period=atr_period)
    df["body_size"] = calculate_body_sizes(df)
    df["avg_body_size"] = calculate_avg_body_size(df, period=20)
    df["upper_wick"] = df.apply(calculate_upper_wick, axis=1)
    df["lower_wick"] = df.apply(calculate_lower_wick, axis=1)
    df["is_bullish"] = df["close"] > df["open"]

    return df
