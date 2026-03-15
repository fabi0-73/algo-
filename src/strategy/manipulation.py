"""
Phase 2: Manipulation Detection (Fake Breakout / Stop Hunt)
Identifies false breakouts that sweep liquidity and reverse.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from config import STRATEGY, TIME_CONFIG
from .consolidation import ConsolidationResult
from .indicators import calculate_atr


def _est_midnight_hour_in_data_tz() -> int:
    """Return the hour in the data timezone that corresponds to 00:00 EST (05:00 UTC).

    For UTC data this returns 5.  For Europe/Athens (UTC+2) this returns 7.
    Uses a fixed winter-time offset (no DST adjustment) since the trading
    calendar midnight is conventionally defined as NY 00:00 EST.
    """
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    data_tz_name = TIME_CONFIG.get("data_timezone", "UTC")
    if data_tz_name == "UTC":
        return 5

    # Use a fixed winter date to get a stable offset
    reference = datetime(2024, 1, 15, 5, 0, tzinfo=timezone.utc)
    local = reference.astimezone(ZoneInfo(data_tz_name))
    return local.hour


# Cache the result so we only compute once
_MIDNIGHT_HOUR = _est_midnight_hour_in_data_tz()


@dataclass
class ManipulationResult:
    """Result of manipulation (fake breakout) detection."""
    valid: bool
    direction: str = ""  # "UP" (fakeout above) or "DOWN" (fakeout below)
    extreme_price: float = 0.0  # Highest/lowest point of the fakeout
    break_distance: float = 0.0  # How far price broke beyond range
    return_candle_idx: int = 0  # Index of candle that returned to range
    atr: float = 0.0
    # Liquidity sweep confirmation
    swept_liquidity: bool = False  # Whether manipulation swept prior swing
    swept_level: float = 0.0       # The swing level that was swept
    swept_swing_idx: int = 0       # Index of the swept swing point
    # Volume spike during manipulation (stop hunt creates volume)
    volume_confirmed: bool = False
    volume_ratio: float = 0.0      # Max vol during manip / baseline
    # Judas swing quality fields
    manipulation_candle_count: int = 0  # Candles from break to return
    velocity_score: float = 0.0         # Break distance per candle / ATR
    session_score: float = 0.0          # Higher during London open
    midnight_price_swept: bool = False  # Swept above/below midnight open
    judas_quality: int = 0              # Composite quality score (0-3)

    @property
    def is_bullish_setup(self) -> bool:
        """Fakeout DOWN = bullish setup (expecting upward distribution)."""
        return self.direction == "DOWN"
    
    @property
    def is_bearish_setup(self) -> bool:
        """Fakeout UP = bearish setup (expecting downward distribution)."""
        return self.direction == "UP"


def detect_manipulation(
    df: pd.DataFrame,
    consolidation: ConsolidationResult,
    break_atr_mult: float = None,
    max_return_candles: int = None,
) -> ManipulationResult:
    """
    Detect if a valid manipulation (fake breakout) occurred after consolidation.
    
    Manipulation criteria:
    1. Price breaks beyond range by >= break_atr_mult * ATR(14)
    2. Price returns back into range within max_return_candles
    
    Args:
        df: DataFrame with OHLC data (starting from after consolidation)
        consolidation: The consolidation zone to check breakout from
        break_atr_mult: Minimum break distance as ATR multiple (default from config)
        max_return_candles: Max candles to return inside (default from config)
    
    Returns:
        ManipulationResult with detection results
    """
    # Use config defaults
    break_atr_mult = break_atr_mult or STRATEGY["manipulation_break_atr_mult"]
    max_return_candles = max_return_candles or STRATEGY["manipulation_return_candles"]
    
    if not consolidation.valid:
        return ManipulationResult(valid=False)
    
    # Need candles after consolidation
    start_idx = consolidation.end_idx + 1
    if start_idx >= len(df):
        return ManipulationResult(valid=False)
    
    # Look at candles after consolidation (within reasonable window)
    search_window = min(max_return_candles + 5, len(df) - start_idx)
    if search_window < 2:
        return ManipulationResult(valid=False)
    
    post_consol = df.iloc[start_idx:start_idx + search_window]
    
    # Get ATR
    atr = consolidation.atr
    min_break_distance = break_atr_mult * atr
    
    range_high = consolidation.range_high
    range_low = consolidation.range_low
    
    # Check for upward fakeout (break above, then return)
    upward_result = _check_upward_fakeout(
        post_consol, range_high, range_low, min_break_distance, max_return_candles, atr, start_idx
    )
    if upward_result.valid:
        return upward_result
    
    # Check for downward fakeout (break below, then return)
    downward_result = _check_downward_fakeout(
        post_consol, range_high, range_low, min_break_distance, max_return_candles, atr, start_idx
    )
    if downward_result.valid:
        return downward_result
    
    return ManipulationResult(valid=False, atr=atr)


def _check_upward_fakeout(
    candles: pd.DataFrame,
    range_high: float,
    range_low: float,
    min_break_distance: float,
    max_return_candles: int,
    atr: float,
    start_idx: int,
) -> ManipulationResult:
    """Check for upward fake breakout (break above then return)."""
    
    extreme_high = 0.0
    break_candle_idx = -1
    
    for i, (idx, candle) in enumerate(candles.iterrows()):
        # Check if this candle breaks above
        if candle["high"] > range_high + min_break_distance:
            if candle["high"] > extreme_high:
                extreme_high = candle["high"]
                if break_candle_idx == -1:
                    break_candle_idx = i
        
        # If we found a break, check if price returns
        if break_candle_idx >= 0:
            candles_since_break = i - break_candle_idx
            
            # Check if close is back inside range
            if candle["close"] <= range_high and candle["close"] >= range_low:
                if candles_since_break <= max_return_candles:
                    return ManipulationResult(
                        valid=True,
                        direction="UP",
                        extreme_price=extreme_high,
                        break_distance=extreme_high - range_high,
                        return_candle_idx=start_idx + i,
                        atr=atr,
                    )
            
            # Too many candles without returning
            if candles_since_break > max_return_candles:
                break
    
    return ManipulationResult(valid=False, atr=atr)


def _check_downward_fakeout(
    candles: pd.DataFrame,
    range_high: float,
    range_low: float,
    min_break_distance: float,
    max_return_candles: int,
    atr: float,
    start_idx: int,
) -> ManipulationResult:
    """Check for downward fake breakout (break below then return)."""
    
    extreme_low = float("inf")
    break_candle_idx = -1
    
    for i, (idx, candle) in enumerate(candles.iterrows()):
        # Check if this candle breaks below
        if candle["low"] < range_low - min_break_distance:
            if candle["low"] < extreme_low:
                extreme_low = candle["low"]
                if break_candle_idx == -1:
                    break_candle_idx = i
        
        # If we found a break, check if price returns
        if break_candle_idx >= 0:
            candles_since_break = i - break_candle_idx
            
            # Check if close is back inside range
            if candle["close"] >= range_low and candle["close"] <= range_high:
                if candles_since_break <= max_return_candles:
                    return ManipulationResult(
                        valid=True,
                        direction="DOWN",
                        extreme_price=extreme_low,
                        break_distance=range_low - extreme_low,
                        return_candle_idx=start_idx + i,
                        atr=atr,
                    )
            
            # Too many candles without returning
            if candles_since_break > max_return_candles:
                break
    
    return ManipulationResult(valid=False, atr=atr)


def scan_for_manipulation(
    df: pd.DataFrame,
    consolidation: ConsolidationResult,
    max_candles_to_scan: int = 20,
) -> ManipulationResult:
    """
    Scan forward from consolidation to find manipulation.
    
    This is useful when scanning historical data where the manipulation
    might have occurred some candles after consolidation ended.
    
    Args:
        df: Full DataFrame
        consolidation: The consolidation result
        max_candles_to_scan: Maximum candles to scan forward
    
    Returns:
        ManipulationResult if found
    """
    if not consolidation.valid:
        return ManipulationResult(valid=False)
    
    end_idx = min(consolidation.end_idx + max_candles_to_scan, len(df))
    scan_df = df.iloc[:end_idx]
    
    return detect_manipulation(scan_df, consolidation)


def confirm_liquidity_sweep(
    df: pd.DataFrame,
    manip: ManipulationResult,
    lookback: int = None,
) -> ManipulationResult:
    """
    Confirm that manipulation swept a prior swing high/low (liquidity).

    For DOWN manipulation: Should have swept a prior swing low
    For UP manipulation: Should have swept a prior swing high

    This confirms the fakeout actually grabbed stops at an obvious level,
    which is a key characteristic of true smart money manipulation.

    Args:
        df: DataFrame with OHLC data
        manip: ManipulationResult to validate
        lookback: How far back to search for swing points (default from config)

    Returns:
        Updated ManipulationResult with liquidity sweep info
    """
    from .market_structure import find_recent_swing_high, find_recent_swing_low

    if not manip.valid:
        return manip

    lookback = lookback or STRATEGY.get("liquidity_sweep_lookback", 50)

    # Search for swings BEFORE the manipulation started
    # The manipulation return_candle_idx marks when price came back
    # Subtract manipulation_return_candles to estimate when break started
    return_candles = STRATEGY.get("manipulation_return_candles", 8)
    search_end_idx = max(0, manip.return_candle_idx - return_candles)

    if manip.direction == "DOWN":
        # Manipulation swept down - should have taken out a prior swing low
        swing = find_recent_swing_low(
            df,
            current_idx=search_end_idx,
            lookback=lookback,
            swing_strength=3
        )

        if swing and swing.valid and manip.extreme_price < swing.price:
            # Manipulation went below the swing low = swept liquidity
            manip.swept_liquidity = True
            manip.swept_level = swing.price
            manip.swept_swing_idx = swing.candle_idx
    else:
        # Manipulation swept up - should have taken out a prior swing high
        swing = find_recent_swing_high(
            df,
            current_idx=search_end_idx,
            lookback=lookback,
            swing_strength=3
        )

        if swing and swing.valid and manip.extreme_price > swing.price:
            # Manipulation went above the swing high = swept liquidity
            manip.swept_liquidity = True
            manip.swept_level = swing.price
            manip.swept_swing_idx = swing.candle_idx

    return manip


def confirm_volume_spike(
    df: pd.DataFrame,
    manip: ManipulationResult,
    volume_ma_period: int = None,
    spike_ratio: float = None,
) -> ManipulationResult:
    """
    Confirm manipulation had elevated volume (stop triggers create volume).

    Args:
        df: DataFrame with OHLC and tick_volume data
        manip: ManipulationResult to validate
        volume_ma_period: Period for volume MA baseline (default from config)
        spike_ratio: Min ratio of manipulation vol to baseline (default from config)

    Returns:
        Updated ManipulationResult with volume_confirmed and volume_ratio set
    """
    if not manip.valid:
        return manip

    if "tick_volume" not in df.columns:
        return manip

    volume_ma_period = volume_ma_period or 20
    spike_ratio = spike_ratio or STRATEGY.get("manipulation_volume_spike_ratio", 1.5)
    return_candles = STRATEGY.get("manipulation_return_candles", 12)

    manip_start = max(0, manip.return_candle_idx - return_candles)
    manip_end = min(manip.return_candle_idx + 1, len(df))

    if manip_end <= manip_start or manip_end > len(df):
        return manip

    vol_ma = df["tick_volume"].rolling(volume_ma_period).mean()
    manip_vol = df["tick_volume"].iloc[manip_start:manip_end].max()

    baseline_idx = manip_start - 1
    if baseline_idx < volume_ma_period:
        return manip
    baseline_vol = vol_ma.iloc[baseline_idx]
    if pd.isna(baseline_vol) or baseline_vol <= 0:
        return manip

    if manip_vol >= baseline_vol * spike_ratio:
        manip.volume_confirmed = True
        manip.volume_ratio = float(manip_vol / baseline_vol)

    return manip


def get_midnight_open_fast(
    midnight_opens: dict, timestamps_arr, current_idx: int
) -> Optional[float]:
    """O(1) midnight open lookup using pre-computed dict."""
    ts = pd.Timestamp(timestamps_arr[current_idx])
    if hasattr(ts, "hour"):
        target_date = ts.date() if ts.hour >= _MIDNIGHT_HOUR else (ts - pd.Timedelta(days=1)).date()
        return midnight_opens.get(target_date)
    return None


def get_midnight_open(df: pd.DataFrame, current_idx: int) -> Optional[float]:
    """
    Get the midnight (00:00 EST / 05:00 UTC) open price for the trading day
    of the candle at current_idx.

    Returns None if no midnight candle is found.
    """
    if "timestamp" not in df.columns:
        return None

    current_ts = df.iloc[current_idx].get("timestamp")
    if current_ts is None:
        return None

    if hasattr(current_ts, "hour"):
        target_hour = _MIDNIGHT_HOUR  # 00:00 EST mapped to data timezone
        current_date = current_ts.date() if current_ts.hour >= target_hour else (current_ts - pd.Timedelta(days=1)).date()

        for i in range(current_idx, max(current_idx - 300, -1), -1):
            ts = df.iloc[i].get("timestamp")
            if ts is None:
                continue
            if ts.date() == current_date and ts.hour == target_hour and ts.minute == 0:
                return float(df.iloc[i]["open"])
            if ts.date() < current_date:
                break

    return None


def score_judas_quality(
    df: pd.DataFrame,
    manip: ManipulationResult,
    break_candle_idx: int = -1,
) -> ManipulationResult:
    """
    Score manipulation quality as a Judas Swing.

    Quality factors:
    1. Candle count: Fast sweeps (1-3 candles) score +1
    2. Velocity: Break distance per candle >= judas_min_velocity_atr * ATR scores +1
    3. Session: Manipulation during London open (07:00-10:00 UTC) scores +1
    4. Midnight sweep: Price swept above/below midnight open scores +1 (bonus)

    Args:
        df: DataFrame with OHLC and timestamp data
        manip: ManipulationResult to score
        break_candle_idx: Index of the first break candle (-1 = estimate)

    Returns:
        ManipulationResult with judas quality fields populated
    """
    if not manip.valid:
        return manip

    max_quality_candles = STRATEGY.get("judas_max_candles", 5)
    min_velocity_atr = STRATEGY.get("judas_min_velocity_atr", 0.3)
    atr = manip.atr if manip.atr > 0 else (manip.break_distance or 1.0)

    # Estimate break candle if not provided
    if break_candle_idx < 0:
        return_candles = STRATEGY.get("manipulation_return_candles", 12)
        break_candle_idx = max(0, manip.return_candle_idx - return_candles)

    candle_count = manip.return_candle_idx - break_candle_idx
    manip.manipulation_candle_count = max(1, candle_count)

    # Velocity: break distance per candle normalised by ATR
    if manip.manipulation_candle_count > 0 and atr > 0:
        manip.velocity_score = (manip.break_distance / manip.manipulation_candle_count) / atr
    else:
        manip.velocity_score = 0.0

    # Composite quality
    quality = 0

    if manip.manipulation_candle_count <= max_quality_candles:
        quality += 1

    if manip.velocity_score >= min_velocity_atr:
        quality += 1

    # Session scoring: London open window (configurable, default 07:00-10:00 UTC)
    london_start = STRATEGY.get("judas_london_start_hour", 7)
    london_end = STRATEGY.get("judas_london_end_hour", 10)
    if STRATEGY.get("judas_london_bonus", True) and "timestamp" in df.columns:
        ts = df.iloc[min(manip.return_candle_idx, len(df) - 1)].get("timestamp")
        if ts is not None and hasattr(ts, "hour"):
            if london_start <= ts.hour <= london_end:
                manip.session_score = 1.0
                quality += 1

    # Midnight price sweep
    midnight_open = get_midnight_open(df, manip.return_candle_idx)
    if midnight_open is not None:
        if manip.direction == "UP" and manip.extreme_price > midnight_open:
            manip.midnight_price_swept = True
        elif manip.direction == "DOWN" and manip.extreme_price < midnight_open:
            manip.midnight_price_swept = True

    manip.judas_quality = quality
    return manip


def score_judas_quality_fast(
    timestamps_arr,
    midnight_opens: dict,
    manip: ManipulationResult,
    break_candle_idx: int = -1,
) -> ManipulationResult:
    """Same scoring logic as score_judas_quality but using pre-extracted arrays."""
    if not manip.valid:
        return manip

    max_quality_candles = STRATEGY.get("judas_max_candles", 5)
    min_velocity_atr = STRATEGY.get("judas_min_velocity_atr", 0.3)
    atr = manip.atr if manip.atr > 0 else (manip.break_distance or 1.0)

    if break_candle_idx < 0:
        return_candles = STRATEGY.get("manipulation_return_candles", 12)
        break_candle_idx = max(0, manip.return_candle_idx - return_candles)

    candle_count = manip.return_candle_idx - break_candle_idx
    manip.manipulation_candle_count = max(1, candle_count)

    if manip.manipulation_candle_count > 0 and atr > 0:
        manip.velocity_score = (manip.break_distance / manip.manipulation_candle_count) / atr
    else:
        manip.velocity_score = 0.0

    quality = 0

    if manip.manipulation_candle_count <= max_quality_candles:
        quality += 1

    if manip.velocity_score >= min_velocity_atr:
        quality += 1

    if STRATEGY.get("judas_london_bonus", True):
        idx = min(manip.return_candle_idx, len(timestamps_arr) - 1)
        ts = pd.Timestamp(timestamps_arr[idx])
        if hasattr(ts, "hour") and 7 <= ts.hour <= 10:
            manip.session_score = 1.0
            quality += 1

    midnight_open = get_midnight_open_fast(midnight_opens, timestamps_arr, manip.return_candle_idx)
    if midnight_open is not None:
        if manip.direction == "UP" and manip.extreme_price > midnight_open:
            manip.midnight_price_swept = True
        elif manip.direction == "DOWN" and manip.extreme_price < midnight_open:
            manip.midnight_price_swept = True

    manip.judas_quality = quality
    return manip
