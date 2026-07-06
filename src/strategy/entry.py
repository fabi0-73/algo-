"""
Phase 4: Entry Logic
Determines optimal entry point on retest with rejection confirmation.
Supports multiple entry modes with SMC confluence (FVG, Order Block, BOS).
"""
from dataclasses import dataclass, field
from typing import Optional, List
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from config import STRATEGY
from .consolidation import ConsolidationResult
from .manipulation import ManipulationResult
from .distribution import DistributionResult
from .indicators import is_rejection_candle, calculate_body_size
from .fvg import FVG, find_fvg_at_retest_level, is_price_leaving_fvg, find_fvgs_in_range
from .order_blocks import (
    OrderBlock,
    BreakerBlock,
    find_ob_at_retest_level,
    find_breaker_at_retest_level,
    is_price_at_order_block,
    find_order_blocks_in_range,
)
from .market_structure import StructureBreak, find_bos_after_manipulation
from .fib import is_in_ote, ote_entry_price


# Entry mode constants
ENTRY_MODE_RETEST_ONLY = "RETEST_ONLY"
ENTRY_MODE_RETEST_WITH_FVG = "RETEST_WITH_FVG"
ENTRY_MODE_ORDER_BLOCK = "ORDER_BLOCK"
ENTRY_MODE_PEAK_LOW = "PEAK_LOW"


# =============================================================================
# Premium/Discount Zone Helpers (SMC Improvement)
# =============================================================================

def is_in_discount_zone(price: float, range_high: float, range_low: float) -> bool:
    """
    Check if price is in discount zone (below 50% of range).

    In SMC, you want to buy in discount (below equilibrium) for better R:R.

    Args:
        price: Current price to check
        range_high: High of the consolidation range
        range_low: Low of the consolidation range

    Returns:
        True if price is in discount zone (below midpoint)
    """
    if range_high <= range_low:
        return False
    midpoint = (range_high + range_low) / 2
    return price < midpoint


def is_in_premium_zone(price: float, range_high: float, range_low: float) -> bool:
    """
    Check if price is in premium zone (above 50% of range).

    In SMC, you want to sell in premium (above equilibrium) for better R:R.

    Args:
        price: Current price to check
        range_high: High of the consolidation range
        range_low: Low of the consolidation range

    Returns:
        True if price is in premium zone (above midpoint)
    """
    if range_high <= range_low:
        return False
    midpoint = (range_high + range_low) / 2
    return price > midpoint


def _equal_level_swept(consolidation: ConsolidationResult, manip: ManipulationResult, atr: float = 1.0) -> bool:
    """True if manipulation swept an equal high or equal low (liquidity pool).

    Uses ATR-based tolerance to match how equal levels are detected in
    consolidation (equal_level_tolerance_atr_mult * ATR).
    """
    tolerance = atr * 0.05  # Match consolidation detection tolerance
    if manip.direction == "UP":
        return bool(
            consolidation.has_equal_highs
            and consolidation.equal_high_level > 0
            and manip.extreme_price >= consolidation.equal_high_level - tolerance
        )
    if manip.direction == "DOWN":
        return bool(
            consolidation.has_equal_lows
            and consolidation.equal_low_level > 0
            and manip.extreme_price <= consolidation.equal_low_level + tolerance
        )
    return False


def calculate_confluence_score(
    bos_confirmed: bool,
    fvg_at_level: bool,
    ob_at_level: bool,
    equal_level_swept: bool,
    volume_confirmed: bool,
    breaker_confluence: bool = False,
    ote_confluence: bool = False,
) -> int:
    """Calculate institutional confluence score from SMC factors.

    Returns a score 0-7 based on how many factors are present:
    BOS (+1), FVG (+1), OB (+1), equal level swept (+1),
    volume confirmed (+1), breaker block (+1), retest in OTE zone (+1).
    """
    score = 0
    if bos_confirmed:
        score += 1
    if fvg_at_level:
        score += 1
    if ob_at_level:
        score += 1
    if equal_level_swept:
        score += 1
    if volume_confirmed:
        score += 1
    if breaker_confluence:
        score += 1
    if ote_confluence:
        score += 1
    return score


def calculate_move_potential(
    velocity_score: float = 0.0,
    session_hour: int = -1,
    body_expansion: float = 0.0,
    consolidation_quality: int = 0,
    equal_level_swept: bool = False,
) -> int:
    """Score move potential 0-5. Higher = pattern predicts bigger price move.

    Separate from confluence_score (which measures entry validity).
    This measures exit expectation — how far the setup is likely to run.

    Factors:
    - velocity_score >= 0.5: strong institutional sweep (+1)
    - London 07-10 or NY 13-15 kill zone: institutional volume hours (+1)
    - body_expansion >= 1.5x: strong distribution follow-through (+1)
    - consolidation_quality >= 2: tight, clean accumulation (+1)
    - equal_level_swept: liquidity pool run (+1)
    """
    score = 0
    if velocity_score >= 0.5:
        score += 1
    if 7 <= session_hour <= 10 or 13 <= session_hour <= 15:
        score += 1
    if body_expansion >= 1.5:
        score += 1
    if consolidation_quality >= 2:
        score += 1
    if equal_level_swept:
        score += 1
    return score


# Confidence-label thresholds for calculate_signal_confidence's 0-4 score
CONFIDENCE_LABELS = [(4, "HIGH"), (3, "GOOD"), (2, "MODERATE"), (0, "LOW")]


def calculate_signal_confidence(
    confluence_score: int,
    move_potential: int,
    entry_hour: int,
) -> tuple:
    """Empirical per-trade confidence score (0-4) and label.

    Calibrated on 280 honest-cost backtest trades (runs ff8c3c7e + f99ef66e,
    2024-09..2026-02) and verified to hold on the most recent 30% window:
      LOW (0-1): 36.8% WR, 0.106R | MODERATE (2): 46.7%, 0.304R
      GOOD (3):  46.8% WR, 0.543R | HIGH (4):     53.5%, 0.581R
    Factors (weights from that data, largest effect first):
      +2 prime hours 13-17 (broker frame; avg 0.55R vs 0.19R off-hours)
      +1 confluence_score >= 4  (score 5 showed NO extra edge — capped)
      +1 move_potential >= 3

    Returns (score, label). Display/sizing aid only — never a hard entry gate
    (LOW trades are net-positive and cushion the equity curve; do not skip them).
    """
    score = 0
    if 13 <= entry_hour <= 17:
        score += 2
    if confluence_score >= 4:
        score += 1
    if move_potential >= 3:
        score += 1
    for threshold, label in CONFIDENCE_LABELS:
        if score >= threshold:
            return score, label
    return score, "LOW"


def check_premium_discount_filter(
    entry_price: float,
    range_high: float,
    range_low: float,
    direction: str,
) -> tuple:
    """
    Check if entry price meets premium/discount zone requirements.

    Args:
        entry_price: Proposed entry price
        range_high: High of the consolidation range
        range_low: Low of the consolidation range
        direction: "LONG" or "SHORT"

    Returns:
        Tuple of (passes_filter: bool, reason: str)
    """
    require_discount_for_long = STRATEGY.get("require_discount_for_long", False)
    require_premium_for_short = STRATEGY.get("require_premium_for_short", False)

    if direction == "LONG" and require_discount_for_long:
        if not is_in_discount_zone(entry_price, range_high, range_low):
            return False, "long_not_in_discount"

    if direction == "SHORT" and require_premium_for_short:
        if not is_in_premium_zone(entry_price, range_high, range_low):
            return False, "short_not_in_premium"

    return True, ""


@dataclass
class EntrySignal:
    """Entry signal with trade parameters."""
    valid: bool
    direction: str = ""  # "LONG" or "SHORT"
    entry_price: float = 0.0  # Signal price (for compatibility/close)
    entry_candle_idx: int = 0
    entry_timestamp: pd.Timestamp = None
    rejection_confirmed: bool = False
    retest_level: float = 0.0  # The level being retested

    # Context from previous phases
    consolidation_high: float = 0.0
    consolidation_low: float = 0.0
    manipulation_extreme: float = 0.0
    manipulation_direction: str = ""

    # Confluence data
    entry_mode: str = ""           # Which entry mode triggered
    fvg_confluence: bool = False   # FVG at entry
    ob_confluence: bool = False    # Order Block at entry
    breaker_confluence: bool = False  # Breaker Block at entry
    bos_confirmed: bool = False    # Break of Structure confirmed
    equal_level_swept: bool = False  # Manipulation swept equal highs/lows
    volume_confirmed: bool = False   # Volume spike during manipulation
    ote_confluence: bool = False   # Retest level sits in the OTE (deep pullback) band
    confluence_score: int = 0      # Number of confluence factors (0-7)

    # Limit fill parameters
    desired_entry_price: float = 0.0  # Limit order price for fill simulation
    desired_entry_type: str = "LIMIT"  # "LIMIT" or "MARKET"
    desired_entry_model: str = ""  # RETEST, FVG, OB, etc.
    atr: float = 0.0  # ATR at entry for risk calculations

    def __post_init__(self):
        """Set defaults for derived fields."""
        if self.desired_entry_price == 0.0 and self.retest_level > 0:
            # For limit fills, default to retest level
            object.__setattr__(self, 'desired_entry_price', self.retest_level)
        elif self.desired_entry_price == 0.0 and self.entry_price > 0:
            object.__setattr__(self, 'desired_entry_price', self.entry_price)
        if not self.desired_entry_model and self.entry_mode:
            object.__setattr__(self, 'desired_entry_model', self.entry_mode)


def check_entry(
    df: pd.DataFrame,
    consolidation: ConsolidationResult,
    manipulation: ManipulationResult,
    distribution: DistributionResult,
    retest_tolerance_atr_mult: float = None,
    rejection_wick_ratio: float = None,
) -> EntrySignal:
    """
    Check for valid entry after distribution.
    
    Entry criteria:
    1. Wait for price to retest the broken range boundary
    2. Look for rejection candle at the retest level
    
    Args:
        df: DataFrame with OHLC data
        consolidation: Consolidation zone result
        manipulation: Manipulation result
        distribution: Distribution result
        retest_tolerance_atr_mult: How close price must get to boundary (default from config)
        rejection_wick_ratio: Wick to body ratio for rejection (default from config)
    
    Returns:
        EntrySignal with entry details
    """
    # Use config defaults
    retest_tolerance_atr_mult = retest_tolerance_atr_mult or STRATEGY["retest_tolerance_atr_mult"]
    rejection_wick_ratio = rejection_wick_ratio or STRATEGY["rejection_wick_ratio"]
    
    if not all([consolidation.valid, manipulation.valid, distribution.valid]):
        return EntrySignal(valid=False)
    
    # Determine entry parameters based on distribution direction
    if distribution.direction == "UP":
        # LONG setup: retest of range_high from above
        direction = "LONG"
        retest_level = consolidation.range_high
        expected_rejection = "UP"  # Bullish rejection (lower wick)
    else:
        # SHORT setup: retest of range_low from below
        direction = "SHORT"
        retest_level = consolidation.range_low
        expected_rejection = "DOWN"  # Bearish rejection (upper wick)
    
    # Start looking after distribution breakout
    start_idx = distribution.break_candle_idx + 1
    if start_idx >= len(df):
        return EntrySignal(valid=False)
    
    # Look for retest in next N candles
    search_window = min(30, len(df) - start_idx)
    if search_window < 1:
        return EntrySignal(valid=False)
    
    atr = distribution.atr
    tolerance = retest_tolerance_atr_mult * atr
    
    post_dist = df.iloc[start_idx:start_idx + search_window]
    
    for i, (idx, candle) in enumerate(post_dist.iterrows()):
        candle_idx = start_idx + i
        
        if direction == "LONG":
            # Check if low retests range_high (from above)
            distance_to_level = candle["low"] - retest_level
            
            # Price should come down to or slightly below the level
            if distance_to_level <= tolerance and distance_to_level >= -tolerance:
                # Check for rejection
                if is_rejection_candle(candle, expected_rejection, rejection_wick_ratio):
                    # Valid entry signal
                    entry_execution = STRATEGY.get("entry_execution", "LIMIT").upper()
                    desired_price = candle["close"] if entry_execution == "MARKET" else retest_level
                    desired_type = "MARKET" if entry_execution == "MARKET" else "LIMIT"
                    return EntrySignal(
                        valid=True,
                        direction=direction,
                        entry_price=candle["close"],
                        entry_candle_idx=candle_idx,
                        entry_timestamp=candle.get("timestamp"),
                        rejection_confirmed=True,
                        retest_level=retest_level,
                        consolidation_high=consolidation.range_high,
                        consolidation_low=consolidation.range_low,
                        manipulation_extreme=manipulation.extreme_price,
                        manipulation_direction=manipulation.direction,
                        desired_entry_price=desired_price,
                        desired_entry_type=desired_type,
                        desired_entry_model=STRATEGY.get("entry_mode", ENTRY_MODE_RETEST_ONLY),
                    )
        else:
            # Check if high retests range_low (from below)
            distance_to_level = retest_level - candle["high"]
            
            # Price should come up to or slightly above the level
            if distance_to_level <= tolerance and distance_to_level >= -tolerance:
                # Check for rejection
                if is_rejection_candle(candle, expected_rejection, rejection_wick_ratio):
                    # Valid entry signal
                    entry_execution = STRATEGY.get("entry_execution", "LIMIT").upper()
                    desired_price = candle["close"] if entry_execution == "MARKET" else retest_level
                    desired_type = "MARKET" if entry_execution == "MARKET" else "LIMIT"
                    return EntrySignal(
                        valid=True,
                        direction=direction,
                        entry_price=candle["close"],
                        entry_candle_idx=candle_idx,
                        entry_timestamp=candle.get("timestamp"),
                        rejection_confirmed=True,
                        retest_level=retest_level,
                        consolidation_high=consolidation.range_high,
                        consolidation_low=consolidation.range_low,
                        manipulation_extreme=manipulation.extreme_price,
                        manipulation_direction=manipulation.direction,
                        desired_entry_price=desired_price,
                        desired_entry_type=desired_type,
                        desired_entry_model=STRATEGY.get("entry_mode", ENTRY_MODE_RETEST_ONLY),
                    )
    
    return EntrySignal(valid=False)


def check_immediate_entry(
    df: pd.DataFrame,
    consolidation: ConsolidationResult,
    manipulation: ManipulationResult,
    distribution: DistributionResult,
) -> EntrySignal:
    """
    Check for immediate entry on distribution candle if it also retests.
    Sometimes the distribution candle itself can be a valid entry.
    
    Args:
        df: DataFrame with OHLC data
        consolidation: Consolidation zone result
        manipulation: Manipulation result
        distribution: Distribution result
    
    Returns:
        EntrySignal if immediate entry valid
    """
    if not all([consolidation.valid, manipulation.valid, distribution.valid]):
        return EntrySignal(valid=False)
    
    dist_candle_idx = distribution.break_candle_idx
    if dist_candle_idx >= len(df):
        return EntrySignal(valid=False)
    
    candle = df.iloc[dist_candle_idx]
    
    if distribution.direction == "UP":
        direction = "LONG"
        retest_level = consolidation.range_high
        
        # Check if the distribution candle retested range_high
        if candle["low"] <= retest_level:
            entry_execution = STRATEGY.get("entry_execution", "LIMIT").upper()
            desired_price = candle["close"] if entry_execution == "MARKET" else retest_level
            desired_type = "MARKET" if entry_execution == "MARKET" else "LIMIT"
            return EntrySignal(
                valid=True,
                direction=direction,
                entry_price=candle["close"],
                entry_candle_idx=dist_candle_idx,
                entry_timestamp=candle.get("timestamp"),
                rejection_confirmed=False,  # Aggressive entry
                retest_level=retest_level,
                consolidation_high=consolidation.range_high,
                consolidation_low=consolidation.range_low,
                manipulation_extreme=manipulation.extreme_price,
                manipulation_direction=manipulation.direction,
                desired_entry_price=desired_price,
                desired_entry_type=desired_type,
                desired_entry_model=STRATEGY.get("entry_mode", ENTRY_MODE_RETEST_ONLY),
            )
    else:
        direction = "SHORT"
        retest_level = consolidation.range_low
        
        # Check if the distribution candle retested range_low
        if candle["high"] >= retest_level:
            entry_execution = STRATEGY.get("entry_execution", "LIMIT").upper()
            desired_price = candle["close"] if entry_execution == "MARKET" else retest_level
            desired_type = "MARKET" if entry_execution == "MARKET" else "LIMIT"
            return EntrySignal(
                valid=True,
                direction=direction,
                entry_price=candle["close"],
                entry_candle_idx=dist_candle_idx,
                entry_timestamp=candle.get("timestamp"),
                rejection_confirmed=False,  # Aggressive entry
                retest_level=retest_level,
                consolidation_high=consolidation.range_high,
                consolidation_low=consolidation.range_low,
                manipulation_extreme=manipulation.extreme_price,
                manipulation_direction=manipulation.direction,
                desired_entry_price=desired_price,
                desired_entry_type=desired_type,
                desired_entry_model=STRATEGY.get("entry_mode", ENTRY_MODE_RETEST_ONLY),
            )
    
    return EntrySignal(valid=False)


def check_entry_with_confluence(
    df: pd.DataFrame,
    consolidation: ConsolidationResult,
    manipulation: ManipulationResult,
    distribution: DistributionResult,
    structure_break: StructureBreak = None,
    entry_mode: str = None,
    retest_tolerance_atr_mult: float = None,
    rejection_wick_ratio: float = None,
) -> EntrySignal:
    """
    Check for valid entry with SMC confluence factors.
    
    Entry Modes:
    - RETEST_ONLY: Classic retest + rejection (original behavior)
    - RETEST_WITH_FVG: Enter as price leaves FVG near retest level
    - ORDER_BLOCK: Enter at Order Block with rejection
    - PEAK_LOW: Enter at lowest point of retest (aggressive)
    
    Args:
        df: DataFrame with OHLC data
        consolidation: Consolidation zone result
        manipulation: Manipulation result
        distribution: Distribution result
        structure_break: Optional BOS confirmation
        entry_mode: Entry strategy to use
        retest_tolerance_atr_mult: Tolerance for retest detection
        rejection_wick_ratio: Wick ratio for rejection confirmation
    
    Returns:
        EntrySignal with confluence data
    """
    entry_mode = entry_mode or STRATEGY.get("entry_mode", ENTRY_MODE_RETEST_ONLY)
    entry_execution = STRATEGY.get("entry_execution", "LIMIT").upper()
    retest_tolerance_atr_mult = retest_tolerance_atr_mult or STRATEGY["retest_tolerance_atr_mult"]
    rejection_wick_ratio = rejection_wick_ratio or STRATEGY["rejection_wick_ratio"]
    
    if not all([consolidation.valid, manipulation.valid, distribution.valid]):
        return EntrySignal(valid=False)
    
    # Determine direction and retest level
    if distribution.direction == "UP":
        direction = "LONG"
        retest_level = consolidation.range_high
        expected_rejection = "UP"
        fvg_direction = "BULLISH"
        ob_direction = "BULLISH"
    else:
        direction = "SHORT"
        retest_level = consolidation.range_low
        expected_rejection = "DOWN"
        fvg_direction = "BEARISH"
        ob_direction = "BEARISH"
    
    atr = distribution.atr
    tolerance = retest_tolerance_atr_mult * atr
    
    # Search range for confluence detection
    search_start = consolidation.start_idx
    search_end = min(distribution.break_candle_idx + 30, len(df))
    
    # Detect FVGs in the zone
    fvgs = find_fvgs_in_range(
        df, search_start, search_end,
        direction=fvg_direction, atr=atr
    )
    fvg_at_level = find_fvg_at_retest_level(
        df, retest_level, search_start, search_end,
        fvg_direction, atr, tolerance_mult=0.5
    )
    
    # Detect Order Blocks in the zone
    order_blocks = find_order_blocks_in_range(
        df, search_start, search_end,
        direction=ob_direction, atr=atr
    )
    ob_at_level = find_ob_at_retest_level(
        df, retest_level, search_start, search_end,
        ob_direction, atr, tolerance_mult=0.5
    )
    
    # Check BOS confirmation
    bos_confirmed = structure_break is not None and structure_break.valid
    if STRATEGY.get("bos_required", False) and not bos_confirmed:
        return EntrySignal(valid=False)

    # Confluence from consolidation/manipulation (equal levels swept, volume spike)
    equal_level_swept = _equal_level_swept(consolidation, manipulation, atr=atr)
    volume_confirmed = getattr(manipulation, "volume_confirmed", False)

    # Start looking after distribution breakout
    start_idx = distribution.break_candle_idx + 1
    if start_idx >= len(df):
        return EntrySignal(valid=False)
    
    search_window = min(30, len(df) - start_idx)
    if search_window < 1:
        return EntrySignal(valid=False)
    
    post_dist = df.iloc[start_idx:start_idx + search_window]
    
    # Track peak low/high for PEAK_LOW mode
    peak_price = None
    peak_idx = None
    
    for i, (idx, candle) in enumerate(post_dist.iterrows()):
        candle_idx = start_idx + i
        
        if direction == "LONG":
            distance_to_level = candle["low"] - retest_level
            at_retest = distance_to_level <= tolerance and distance_to_level >= -tolerance
            
            # Track peak low
            if at_retest:
                if peak_price is None or candle["low"] < peak_price:
                    peak_price = candle["low"]
                    peak_idx = candle_idx
        else:
            distance_to_level = retest_level - candle["high"]
            at_retest = distance_to_level <= tolerance and distance_to_level >= -tolerance
            
            # Track peak high
            if at_retest:
                if peak_price is None or candle["high"] > peak_price:
                    peak_price = candle["high"]
                    peak_idx = candle_idx
        
        if not at_retest:
            continue
        
        # Check entry based on mode
        entry_triggered = False
        fvg_confluence = False
        ob_confluence = False

        if entry_mode == ENTRY_MODE_RETEST_WITH_FVG:
            # Check if price is leaving an FVG at retest level
            if fvg_at_level and is_price_leaving_fvg(candle, fvg_at_level):
                entry_triggered = True
                fvg_confluence = True
            elif ob_at_level and is_price_at_order_block(candle, ob_at_level, rejection_required=True):
                entry_triggered = True
                ob_confluence = True
            elif STRATEGY.get("allow_rejection_fallback", False):
                # Optional fallback: retest + rejection without FVG/OB
                if is_rejection_candle(candle, expected_rejection, rejection_wick_ratio):
                    entry_triggered = True

        elif entry_mode == ENTRY_MODE_ORDER_BLOCK:
            # Check if price is at Order Block with rejection
            if ob_at_level and is_price_at_order_block(candle, ob_at_level, rejection_required=True):
                entry_triggered = True
                ob_confluence = True

        elif entry_mode == ENTRY_MODE_RETEST_ONLY:
            # Original behavior: retest + rejection
            if is_rejection_candle(candle, expected_rejection, rejection_wick_ratio):
                entry_triggered = True

        # For PEAK_LOW mode, we continue scanning and enter at the peak
        elif entry_mode == ENTRY_MODE_PEAK_LOW:
            # Will be handled after the loop
            pass

        if entry_triggered:
            # Breaker block at retest level (when enabled)
            breaker_at_level = None
            if STRATEGY.get("use_breaker_blocks", False):
                ob_dir = "BULLISH" if direction == "LONG" else "BEARISH"
                breaker_at_level = find_breaker_at_retest_level(
                    df, retest_level, search_start, search_end,
                    candle_idx, ob_dir, atr, tolerance_mult=0.5
                )
            breaker_confluence = breaker_at_level is not None and breaker_at_level.valid

            # Calculate confluence score using shared function
            fvg_confluence = bool(fvg_at_level)
            ob_confluence = bool(ob_at_level)
            confluence_score = calculate_confluence_score(
                bos_confirmed=bos_confirmed,
                fvg_at_level=fvg_confluence,
                ob_at_level=ob_confluence,
                equal_level_swept=equal_level_swept,
                volume_confirmed=volume_confirmed,
                breaker_confluence=breaker_confluence,
            )

            min_confluence = STRATEGY.get("min_confluence_score", 0)
            if min_confluence > 0 and confluence_score < min_confluence:
                continue  # Skip this entry, try next candle

            desired_price = candle["close"] if entry_execution == "MARKET" else retest_level
            desired_type = "MARKET" if entry_execution == "MARKET" else "LIMIT"
            return EntrySignal(
                valid=True,
                direction=direction,
                entry_price=candle["close"],
                entry_candle_idx=candle_idx,
                entry_timestamp=candle.get("timestamp"),
                rejection_confirmed=is_rejection_candle(candle, expected_rejection, rejection_wick_ratio),
                retest_level=retest_level,
                consolidation_high=consolidation.range_high,
                consolidation_low=consolidation.range_low,
                manipulation_extreme=manipulation.extreme_price,
                manipulation_direction=manipulation.direction,
                entry_mode=entry_mode,
                fvg_confluence=fvg_confluence,
                ob_confluence=ob_confluence,
                breaker_confluence=breaker_confluence,
                bos_confirmed=bos_confirmed,
                equal_level_swept=equal_level_swept,
                volume_confirmed=volume_confirmed,
                confluence_score=confluence_score,
                desired_entry_price=desired_price,
                desired_entry_type=desired_type,
                desired_entry_model=entry_mode,
            )
    
    # Handle PEAK_LOW mode - enter at the extreme of the retest
    if entry_mode == ENTRY_MODE_PEAK_LOW and peak_idx is not None:
        peak_candle = df.iloc[peak_idx]

        # Breaker block at retest level (when enabled)
        breaker_at_level = None
        if STRATEGY.get("use_breaker_blocks", False):
            ob_dir = "BULLISH" if direction == "LONG" else "BEARISH"
            breaker_at_level = find_breaker_at_retest_level(
                df, retest_level, search_start, search_end,
                peak_idx, ob_dir, atr, tolerance_mult=0.5
            )
        breaker_confluence = breaker_at_level is not None and breaker_at_level.valid

        # Calculate confluence
        confluence_score = 0
        fvg_confluence = fvg_at_level is not None
        ob_confluence = ob_at_level is not None
        if bos_confirmed:
            confluence_score += 1
        if fvg_confluence:
            confluence_score += 1
        if ob_confluence:
            confluence_score += 1
        if equal_level_swept:
            confluence_score += 1
        if volume_confirmed:
            confluence_score += 1
        if breaker_confluence:
            confluence_score += 1

        min_confluence = STRATEGY.get("min_confluence_score", 0)
        if min_confluence > 0 and confluence_score < min_confluence:
            return EntrySignal(valid=False)

        desired_price = peak_candle["close"] if entry_execution == "MARKET" else retest_level
        desired_type = "MARKET" if entry_execution == "MARKET" else "LIMIT"
        return EntrySignal(
            valid=True,
            direction=direction,
            entry_price=peak_candle["close"],
            entry_candle_idx=peak_idx,
            entry_timestamp=peak_candle.get("timestamp"),
            rejection_confirmed=False,  # Aggressive entry
            retest_level=retest_level,
            consolidation_high=consolidation.range_high,
            consolidation_low=consolidation.range_low,
            manipulation_extreme=manipulation.extreme_price,
            manipulation_direction=manipulation.direction,
            entry_mode=entry_mode,
            fvg_confluence=fvg_confluence,
            ob_confluence=ob_confluence,
            breaker_confluence=breaker_confluence,
            bos_confirmed=bos_confirmed,
            equal_level_swept=equal_level_swept,
            volume_confirmed=volume_confirmed,
            confluence_score=confluence_score,
            desired_entry_price=desired_price,
            desired_entry_type=desired_type,
            desired_entry_model=entry_mode,
        )

    return EntrySignal(valid=False)


def check_entry_at_candle(
    df: pd.DataFrame,
    current_idx: int,
    consolidation: ConsolidationResult,
    manipulation: ManipulationResult,
    distribution: DistributionResult,
    structure_break: StructureBreak = None,
    fvg_at_level: FVG = None,
    ob_at_level: OrderBlock = None,
    entry_mode: str = None,
) -> EntrySignal:
    """
    Check if current candle provides a valid entry with confluence.
    
    This is the main entry check used by the backtest engine for bar-by-bar evaluation.
    
    Args:
        df: DataFrame with OHLC data
        current_idx: Current candle index
        consolidation: Consolidation zone result
        manipulation: Manipulation result
        distribution: Distribution result
        structure_break: Optional BOS confirmation
        fvg_at_level: Pre-detected FVG near retest level
        ob_at_level: Pre-detected Order Block near retest level
        entry_mode: Entry strategy
    
    Returns:
        EntrySignal if entry conditions met
    """
    entry_mode = entry_mode or STRATEGY.get("entry_mode", ENTRY_MODE_RETEST_ONLY)
    retest_tolerance_atr_mult = STRATEGY["retest_tolerance_atr_mult"]
    rejection_wick_ratio = STRATEGY["rejection_wick_ratio"]
    
    if not all([consolidation.valid, manipulation.valid, distribution.valid]):
        return EntrySignal(valid=False)
    
    if current_idx >= len(df):
        return EntrySignal(valid=False)
    
    candle = df.iloc[current_idx]
    atr = distribution.atr
    tolerance = retest_tolerance_atr_mult * atr
    
    # Determine direction and levels
    if distribution.direction == "UP":
        direction = "LONG"
        retest_level = consolidation.range_high
        expected_rejection = "UP"
        distance_to_level = candle["low"] - retest_level
    else:
        direction = "SHORT"
        retest_level = consolidation.range_low
        expected_rejection = "DOWN"
        distance_to_level = retest_level - candle["high"]
    
    # Check if at retest level
    at_retest = distance_to_level <= tolerance and distance_to_level >= -tolerance
    if not at_retest:
        return EntrySignal(valid=False)
    
    # Check BOS requirement
    bos_confirmed = structure_break is not None and structure_break.valid
    if STRATEGY.get("bos_required", False) and not bos_confirmed:
        return EntrySignal(valid=False)
    
    # Check entry based on mode
    entry_triggered = False
    fvg_confluence = fvg_at_level is not None
    ob_confluence = ob_at_level is not None
    
    if entry_mode == ENTRY_MODE_RETEST_WITH_FVG:
        if fvg_at_level and is_price_leaving_fvg(candle, fvg_at_level):
            entry_triggered = True
        elif ob_at_level and is_price_at_order_block(candle, ob_at_level, rejection_required=True):
            entry_triggered = True
        elif STRATEGY.get("allow_rejection_fallback", False):
            if is_rejection_candle(candle, expected_rejection, rejection_wick_ratio):
                entry_triggered = True

    elif entry_mode == ENTRY_MODE_ORDER_BLOCK:
        if ob_at_level and is_price_at_order_block(candle, ob_at_level, rejection_required=True):
            entry_triggered = True
    
    elif entry_mode == ENTRY_MODE_RETEST_ONLY:
        if is_rejection_candle(candle, expected_rejection, rejection_wick_ratio):
            entry_triggered = True
    
    elif entry_mode == ENTRY_MODE_PEAK_LOW:
        # PEAK_LOW requires cross-bar state tracking to find the true peak,
        # which isn't possible in the bar-by-bar check_entry_at_candle path.
        # Fall back to rejection-based entry instead of blindly triggering.
        logger.warning("PEAK_LOW mode not supported in bar-by-bar entry; falling back to rejection check")
        if is_rejection_candle(candle, expected_rejection, rejection_wick_ratio):
            entry_triggered = True
    
    if not entry_triggered:
        return EntrySignal(valid=False)
    
    # Breaker block at retest level (when enabled)
    breaker_at_level = None
    if STRATEGY.get("use_breaker_blocks", False):
        ob_dir = "BULLISH" if direction == "LONG" else "BEARISH"
        search_end = min(distribution.break_candle_idx + 30, len(df))
        breaker_at_level = find_breaker_at_retest_level(
            df, retest_level, consolidation.start_idx, search_end,
            current_idx, ob_dir, atr, tolerance_mult=0.5
        )
    breaker_confluence = breaker_at_level is not None and breaker_at_level.valid
    equal_level_swept = _equal_level_swept(consolidation, manipulation, atr=atr)
    volume_confirmed = getattr(manipulation, "volume_confirmed", False)

    # OTE geometry: is the retest level a deep pullback (61.8-79%) of the
    # manipulation-extreme -> distribution-break leg? Pure geometry, no fib
    # mysticism — deep pullbacks shorten the stop and widen realized R.
    # Gated off by default: turning this on changes confluence scores and thus
    # sizing tiers on the validated champion config — experiment-only.
    ote_conf = False
    _leg_end = getattr(distribution, "break_price", 0.0) or 0.0
    _leg_start = getattr(manipulation, "extreme_price", 0.0) or 0.0
    if (STRATEGY.get("use_ote_confluence", False)
            and _leg_end and _leg_start and _leg_end != _leg_start):
        ote_conf = is_in_ote(retest_level, _leg_start, _leg_end)

    # Calculate confluence score using shared function
    # Note: Judas quality is a pattern quality metric, not an institutional confluence
    # factor. It is enforced separately via min_judas_quality hard gate.
    confluence_score = calculate_confluence_score(
        bos_confirmed=bos_confirmed,
        fvg_at_level=fvg_confluence,
        ob_at_level=ob_confluence,
        equal_level_swept=equal_level_swept,
        volume_confirmed=volume_confirmed,
        breaker_confluence=breaker_confluence,
        ote_confluence=ote_conf,
    )

    # Apply direction-specific confluence minimums
    min_confluence = STRATEGY.get("min_confluence_score", 0)
    if direction == "SHORT":
        short_min = STRATEGY.get("short_min_confluence_score", min_confluence)
        min_confluence = max(min_confluence, short_min)
    if min_confluence > 0 and confluence_score < min_confluence:
        return EntrySignal(valid=False)

    # Determine desired entry price based on execution preference
    entry_execution = STRATEGY.get("entry_execution", "LIMIT").upper()
    desired_price = candle["close"] if entry_execution == "MARKET" else retest_level
    desired_type = "MARKET" if entry_execution == "MARKET" else "LIMIT"

    # E3 refinement: with entry_price_mode=OTE, place the limit at the OTE band
    # edge when it is a strictly better price than the plain retest level.
    # Trade-off measured by the experiment: wider R per fill vs fewer fills.
    if (desired_type == "LIMIT" and _leg_end and _leg_start
            and STRATEGY.get("entry_price_mode", "RETEST").upper() == "OTE"):
        desired_price = ote_entry_price(direction, _leg_start, _leg_end, retest_level)

    return EntrySignal(
        valid=True,
        direction=direction,
        entry_price=candle["close"],
        entry_candle_idx=current_idx,
        entry_timestamp=candle.get("timestamp"),
        rejection_confirmed=is_rejection_candle(candle, expected_rejection, rejection_wick_ratio),
        retest_level=retest_level,
        consolidation_high=consolidation.range_high,
        consolidation_low=consolidation.range_low,
        manipulation_extreme=manipulation.extreme_price,
        manipulation_direction=manipulation.direction,
        entry_mode=entry_mode,
        fvg_confluence=fvg_confluence,
        ob_confluence=ob_confluence,
        breaker_confluence=breaker_confluence,
        bos_confirmed=bos_confirmed,
        equal_level_swept=equal_level_swept,
        volume_confirmed=volume_confirmed,
        ote_confluence=ote_conf,
        confluence_score=confluence_score,
        desired_entry_price=desired_price,
        desired_entry_type=desired_type,
        desired_entry_model=entry_mode,
        atr=atr,
    )
