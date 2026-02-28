"""
Order Block Detection
Identifies the last opposing candle before an impulsive move (institutional entry points).
"""
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np

from config import STRATEGY


@dataclass
class OrderBlock:
    """Order Block representation."""
    valid: bool
    direction: str = ""      # "BULLISH" or "BEARISH"
    top: float = 0.0         # OB high
    bottom: float = 0.0      # OB low
    candle_idx: int = 0      # Index of the OB candle
    strength: float = 0.0    # Strength based on subsequent displacement
    mitigated: bool = False  # Whether OB has been mitigated (price returned)
    mitigation_idx: int = 0  # Index where OB was mitigated
    
    @property
    def midpoint(self) -> float:
        """Middle of the Order Block zone."""
        return (self.top + self.bottom) / 2
    
    @property
    def size(self) -> float:
        """Size of the Order Block."""
        return self.top - self.bottom
    
    def contains_price(self, price: float) -> bool:
        """Check if a price is within the OB zone."""
        return self.bottom <= price <= self.top
    
    def is_near(self, level: float, tolerance: float) -> bool:
        """Check if OB is near a given price level."""
        return (self.bottom - tolerance <= level <= self.top + tolerance)


def is_bullish_candle(candle: pd.Series) -> bool:
    """Check if candle is bullish."""
    return candle["close"] > candle["open"]


def is_bearish_candle(candle: pd.Series) -> bool:
    """Check if candle is bearish."""
    return candle["close"] < candle["open"]


def get_body_size(candle: pd.Series) -> float:
    """Get absolute body size of candle."""
    return abs(candle["close"] - candle["open"])


def detect_order_block(
    df: pd.DataFrame,
    impulse_start_idx: int,
    direction: str,
    min_body_atr_mult: float = None,
    displacement_mult: float = None,
    atr: float = None,
    lookback: int = 10,
) -> Optional[OrderBlock]:
    """
    Detect an Order Block before an impulsive move.
    
    Order Block Definition:
    - Bullish OB: Last bearish candle before an impulsive bullish move
    - Bearish OB: Last bullish candle before an impulsive bearish move
    
    Args:
        df: DataFrame with OHLC data
        impulse_start_idx: Index where the impulsive move started
        direction: Expected OB direction ("BULLISH" or "BEARISH")
        min_body_atr_mult: Minimum body size for OB candle
        displacement_mult: Required move after OB (body multiple)
        atr: ATR value for validation
        lookback: How many candles back to search for OB
    
    Returns:
        OrderBlock if found, None otherwise
    """
    min_body_atr_mult = min_body_atr_mult or STRATEGY.get("ob_min_body_atr_mult", 0.15)
    displacement_mult = displacement_mult or STRATEGY.get("ob_displacement_mult", 1.5)
    
    if impulse_start_idx < 1 or impulse_start_idx >= len(df):
        return None
    
    search_start = max(0, impulse_start_idx - lookback)
    
    # Find the last opposing candle before impulse
    ob_candle_idx = None
    
    for idx in range(impulse_start_idx - 1, search_start - 1, -1):
        candle = df.iloc[idx]
        
        if direction == "BULLISH":
            # Looking for last bearish candle
            if is_bearish_candle(candle):
                ob_candle_idx = idx
                break
        else:
            # Looking for last bullish candle
            if is_bullish_candle(candle):
                ob_candle_idx = idx
                break
    
    if ob_candle_idx is None:
        return None
    
    ob_candle = df.iloc[ob_candle_idx]
    ob_body = get_body_size(ob_candle)
    
    # Validate minimum body size
    if atr is not None and min_body_atr_mult > 0:
        if ob_body < min_body_atr_mult * atr:
            return None
    
    # Calculate displacement (how far price moved after OB)
    # Look at the move from OB to current impulse position
    impulse_end_idx = min(impulse_start_idx + 5, len(df) - 1)
    
    if direction == "BULLISH":
        # Measure upward displacement
        high_after = df.iloc[ob_candle_idx + 1:impulse_end_idx + 1]["high"].max()
        displacement = high_after - ob_candle["high"]
    else:
        # Measure downward displacement
        low_after = df.iloc[ob_candle_idx + 1:impulse_end_idx + 1]["low"].min()
        displacement = ob_candle["low"] - low_after
    
    # Validate displacement
    min_displacement = ob_body * displacement_mult
    if displacement < min_displacement:
        return None
    
    # Calculate strength (displacement relative to OB size)
    strength = displacement / ob_body if ob_body > 0 else 0

    # Use body only or full range for OB boundaries
    use_body_only = STRATEGY.get("ob_use_body_only", True)

    if use_body_only:
        # Use candle body (open/close) - standard SMC approach
        ob_top = max(ob_candle["open"], ob_candle["close"])
        ob_bottom = min(ob_candle["open"], ob_candle["close"])
    else:
        # Use full candle range (high/low)
        ob_top = ob_candle["high"]
        ob_bottom = ob_candle["low"]

    return OrderBlock(
        valid=True,
        direction=direction,
        top=ob_top,
        bottom=ob_bottom,
        candle_idx=ob_candle_idx,
        strength=round(strength, 2),
    )


def find_order_blocks_in_range(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    direction: str = None,
    atr: float = None,
    min_body_atr_mult: float = None,
    displacement_mult: float = None,
) -> List[OrderBlock]:
    """
    Find all Order Blocks in a given range.
    
    Scans for impulsive moves and identifies OBs before them.
    
    Args:
        df: DataFrame with OHLC data
        start_idx: Starting index
        end_idx: Ending index
        direction: Filter by direction
        atr: ATR value
        min_body_atr_mult: Minimum body size for OB
        displacement_mult: Required displacement after OB
    
    Returns:
        List of OrderBlock objects
    """
    order_blocks = []
    
    # First, identify impulsive candles (large body moves)
    avg_body = abs(df["close"] - df["open"]).iloc[start_idx:end_idx].mean()
    
    for idx in range(start_idx + 1, min(end_idx, len(df))):
        candle = df.iloc[idx]
        body = get_body_size(candle)
        
        # Check if this is an impulsive candle (large body)
        if body < avg_body * 1.5:
            continue
        
        # Determine impulse direction
        if is_bullish_candle(candle):
            impulse_dir = "BULLISH"
        else:
            impulse_dir = "BEARISH"
        
        # Skip if direction filter doesn't match
        if direction is not None and direction != impulse_dir:
            continue
        
        # Look for OB before this impulse
        ob = detect_order_block(
            df=df,
            impulse_start_idx=idx,
            direction=impulse_dir,
            min_body_atr_mult=min_body_atr_mult,
            displacement_mult=displacement_mult,
            atr=atr,
        )
        
        if ob is not None:
            # Avoid duplicates (same OB candle)
            if not any(existing.candle_idx == ob.candle_idx for existing in order_blocks):
                order_blocks.append(ob)
    
    return order_blocks


def check_ob_mitigation(
    ob: OrderBlock,
    df: pd.DataFrame,
    start_idx: int = None,
    end_idx: int = None,
) -> OrderBlock:
    """
    Check if an Order Block has been mitigated (price returned to it).
    
    Args:
        ob: The Order Block to check
        df: DataFrame with OHLC data
        start_idx: Start checking from this index
        end_idx: Stop checking at this index
    
    Returns:
        Updated OrderBlock with mitigation status
    """
    if not ob.valid:
        return ob
    
    start_idx = start_idx or (ob.candle_idx + 1)
    end_idx = end_idx or len(df)
    
    for idx in range(start_idx, end_idx):
        candle = df.iloc[idx]
        
        if ob.direction == "BULLISH":
            # OB mitigated when price trades back down into it
            if candle["low"] <= ob.top:
                ob.mitigated = True
                ob.mitigation_idx = idx
                return ob
        else:
            # OB mitigated when price trades back up into it
            if candle["high"] >= ob.bottom:
                ob.mitigated = True
                ob.mitigation_idx = idx
                return ob
    
    return ob


def find_nearest_order_block(
    order_blocks: List[OrderBlock],
    price_level: float,
    direction: str = None,
    max_distance: float = None,
    unmitigated_only: bool = True,
) -> Optional[OrderBlock]:
    """
    Find the Order Block nearest to a given price level.
    
    Args:
        order_blocks: List of OrderBlocks to search
        price_level: Target price level
        direction: Filter by direction
        max_distance: Maximum distance from price level
        unmitigated_only: Only return unmitigated OBs
    
    Returns:
        Nearest OrderBlock or None
    """
    if not order_blocks:
        return None
    
    candidates = order_blocks
    
    if direction:
        candidates = [ob for ob in candidates if ob.direction == direction]
    
    if unmitigated_only:
        candidates = [ob for ob in candidates if not ob.mitigated]
    
    if not candidates:
        return None
    
    # Sort by distance to price level
    def distance_to_level(ob: OrderBlock) -> float:
        if ob.contains_price(price_level):
            return 0
        return min(abs(ob.top - price_level), abs(ob.bottom - price_level))
    
    candidates.sort(key=distance_to_level)
    nearest = candidates[0]
    
    if max_distance is not None:
        if distance_to_level(nearest) > max_distance:
            return None
    
    return nearest


def find_ob_at_retest_level(
    df: pd.DataFrame,
    retest_level: float,
    search_start_idx: int,
    search_end_idx: int,
    direction: str,
    atr: float,
    tolerance_mult: float = 0.5,
) -> Optional[OrderBlock]:
    """
    Find an Order Block near the retest level for entry.
    
    Args:
        df: DataFrame with OHLC data
        retest_level: The price level being retested
        search_start_idx: Start of search range
        search_end_idx: End of search range
        direction: Expected OB direction
        atr: ATR value
        tolerance_mult: How close OB must be to retest level
    
    Returns:
        OrderBlock if found near retest level
    """
    order_blocks = find_order_blocks_in_range(
        df=df,
        start_idx=search_start_idx,
        end_idx=search_end_idx,
        direction=direction,
        atr=atr,
    )
    
    if not order_blocks:
        return None
    
    tolerance = tolerance_mult * atr
    
    # Find OB that overlaps or is close to retest level
    for ob in order_blocks:
        if ob.is_near(retest_level, tolerance):
            return ob
    
    return None


def is_price_at_order_block(
    candle: pd.Series,
    ob: OrderBlock,
    rejection_required: bool = True,
) -> bool:
    """
    Check if price is at an Order Block (potential entry).
    
    Args:
        candle: Current candle data
        ob: The Order Block
        rejection_required: Whether to require rejection confirmation
    
    Returns:
        True if price is at OB with optional rejection
    """
    if not ob.valid:
        return False
    
    if ob.direction == "BULLISH":
        # Price touching OB from above
        price_at_ob = candle["low"] <= ob.top and candle["low"] >= ob.bottom
        
        if not price_at_ob:
            return False
        
        if rejection_required:
            # Check for bullish rejection (close above OB top)
            return candle["close"] > ob.top
        return True
    else:
        # Price touching OB from below
        price_at_ob = candle["high"] >= ob.bottom and candle["high"] <= ob.top
        
        if not price_at_ob:
            return False
        
        if rejection_required:
            # Check for bearish rejection (close below OB bottom)
            return candle["close"] < ob.bottom
        return True


# =============================================================================
# Breaker Block Detection (SMC Improvement)
# =============================================================================

@dataclass
class BreakerBlock:
    """
    Breaker Block representation.

    A breaker block forms when an Order Block is broken through and
    becomes a support/resistance zone for the opposite direction.

    Example: A bullish OB that gets broken to the downside becomes a
    bearish breaker - price may rally back up to retest it before continuing down.
    """
    valid: bool
    direction: str = ""       # Direction of expected move ("BULLISH" or "BEARISH")
    top: float = 0.0
    bottom: float = 0.0
    original_ob_idx: int = 0  # Index of original OB candle
    break_idx: int = 0        # Index where OB was broken
    formed_from: str = ""     # "bullish_ob_broken" or "bearish_ob_broken"

    @property
    def midpoint(self) -> float:
        """Middle of the breaker zone."""
        return (self.top + self.bottom) / 2

    @property
    def size(self) -> float:
        """Size of the breaker zone."""
        return self.top - self.bottom


def detect_breaker_block(
    df: pd.DataFrame,
    ob: OrderBlock,
    current_idx: int,
    min_break_candles: int = 2,
) -> Optional[BreakerBlock]:
    """
    Detect if an Order Block has become a Breaker Block.

    A Breaker Block forms when:
    1. A bullish OB is broken to the downside -> becomes bearish breaker
    2. A bearish OB is broken to the upside -> becomes bullish breaker

    The breaker is then valid for retest from the opposite direction.

    Args:
        df: DataFrame with OHLC data
        ob: Original OrderBlock to check
        current_idx: Current index (to search for break)
        min_break_candles: Minimum candles closing beyond OB to confirm break

    Returns:
        BreakerBlock if OB was broken, None otherwise
    """
    if not ob.valid:
        return None

    # Search for break after OB formed
    search_start = ob.candle_idx + 1
    search_end = min(current_idx, search_start + 30)

    break_candle_count = 0
    break_idx = 0

    for idx in range(search_start, search_end):
        if idx >= len(df):
            break

        candle = df.iloc[idx]

        if ob.direction == "BULLISH":
            # Bullish OB broken when price closes below OB bottom
            if candle["close"] < ob.bottom:
                if break_candle_count == 0:
                    break_idx = idx
                break_candle_count += 1
            else:
                break_candle_count = 0  # Reset if price comes back
        else:
            # Bearish OB broken when price closes above OB top
            if candle["close"] > ob.top:
                if break_candle_count == 0:
                    break_idx = idx
                break_candle_count += 1
            else:
                break_candle_count = 0

        if break_candle_count >= min_break_candles:
            # OB is broken - becomes breaker with opposite direction
            breaker_direction = "BEARISH" if ob.direction == "BULLISH" else "BULLISH"
            formed_from = f"{ob.direction.lower()}_ob_broken"

            return BreakerBlock(
                valid=True,
                direction=breaker_direction,
                top=ob.top,
                bottom=ob.bottom,
                original_ob_idx=ob.candle_idx,
                break_idx=break_idx,
                formed_from=formed_from,
            )

    return None


def find_breakers_in_range(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    direction: str,
    current_idx: int,
    atr: float = None,
    tolerance_mult: float = 0.5,
) -> List["BreakerBlock"]:
    """
    Find Breaker Blocks in a range that are valid at current_idx.

    For LONG we need BULLISH breakers (bearish OB broken up).
    For SHORT we need BEARISH breakers (bullish OB broken down).

    Args:
        df: DataFrame with OHLC data
        start_idx: Start of search range
        end_idx: End of search range
        direction: Expected move direction ("BULLISH" or "BEARISH")
        current_idx: Current candle index (breakers must be formed by this point)
        atr: ATR for OB detection
        tolerance_mult: Unused, for API consistency

    Returns:
        List of BreakerBlock objects
    """
    # Breaker direction matches entry direction: LONG -> BULLISH breaker, SHORT -> BEARISH breaker
    # BULLISH breaker = bearish OB broken up; BEARISH breaker = bullish OB broken down
    ob_direction = "BEARISH" if direction == "BULLISH" else "BULLISH"
    obs = find_order_blocks_in_range(
        df=df,
        start_idx=start_idx,
        end_idx=end_idx,
        direction=ob_direction,
        atr=atr,
    )
    breakers = []
    for ob in obs:
        bb = detect_breaker_block(df, ob, current_idx)
        if bb is not None and bb.valid and bb.direction == direction:
            breakers.append(bb)
    return breakers


def find_breaker_at_retest_level(
    df: pd.DataFrame,
    retest_level: float,
    search_start_idx: int,
    search_end_idx: int,
    current_idx: int,
    direction: str,
    atr: float,
    tolerance_mult: float = 0.5,
) -> Optional["BreakerBlock"]:
    """
    Find a Breaker Block near the retest level for entry confluence.

    Args:
        df: DataFrame with OHLC data
        retest_level: The price level being retested
        search_start_idx: Start of search range
        search_end_idx: End of search range
        current_idx: Current candle index
        direction: "BULLISH" or "BEARISH"
        atr: ATR value
        tolerance_mult: How close breaker must be to retest level (ATR multiple)

    Returns:
        BreakerBlock if found near retest level, None otherwise
    """
    breakers = find_breakers_in_range(
        df=df,
        start_idx=search_start_idx,
        end_idx=search_end_idx,
        direction=direction,
        current_idx=current_idx,
        atr=atr,
        tolerance_mult=tolerance_mult,
    )
    if not breakers:
        return None
    tolerance = tolerance_mult * atr
    for bb in breakers:
        if bb.bottom - tolerance <= retest_level <= bb.top + tolerance:
            return bb
    return None


def is_price_at_breaker(
    candle: pd.Series,
    breaker: BreakerBlock,
    rejection_required: bool = True,
) -> bool:
    """
    Check if price is retesting a breaker block from the new direction.

    For bullish breaker (former bearish OB broken up): price retests from above
    For bearish breaker (former bullish OB broken down): price retests from below

    Args:
        candle: Current candle data
        breaker: The BreakerBlock to check
        rejection_required: Whether to require rejection confirmation

    Returns:
        True if price is at breaker with expected reaction
    """
    if not breaker.valid:
        return False

    if breaker.direction == "BULLISH":
        # Bullish breaker: price comes down to retest (we expect it to bounce up)
        price_at_breaker = candle["low"] <= breaker.top and candle["low"] >= breaker.bottom

        if not price_at_breaker:
            return False

        if rejection_required:
            return candle["close"] > breaker.top  # Bullish rejection
        return True
    else:
        # Bearish breaker: price comes up to retest (we expect it to drop)
        price_at_breaker = candle["high"] >= breaker.bottom and candle["high"] <= breaker.top

        if not price_at_breaker:
            return False

        if rejection_required:
            return candle["close"] < breaker.bottom  # Bearish rejection
        return True
