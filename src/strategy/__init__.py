"""Strategy components - AMD phases with SMC confluence"""
from .indicators import calculate_atr, calculate_body_sizes
from .consolidation import detect_consolidation, ConsolidationResult
from .manipulation import detect_manipulation, ManipulationResult
from .distribution import detect_distribution, DistributionResult
from .entry import (
    check_entry, EntrySignal, check_entry_with_confluence, check_entry_at_candle,
    ENTRY_MODE_RETEST_ONLY, ENTRY_MODE_RETEST_WITH_FVG,
    ENTRY_MODE_ORDER_BLOCK, ENTRY_MODE_PEAK_LOW
)
from .risk import calculate_risk, RiskParams

# SMC Confluence modules
from .fvg import FVG, detect_fvg, find_fvgs_in_range, find_fvg_at_retest_level
from .order_blocks import OrderBlock, detect_order_block, find_order_blocks_in_range, find_ob_at_retest_level
from .market_structure import (
    SwingPoint, StructureBreak, detect_swing_high, detect_swing_low,
    find_swing_points, find_bos_after_manipulation, detect_break_of_structure
)

__all__ = [
    # Indicators
    "calculate_atr",
    "calculate_body_sizes",
    # AMD Phases
    "detect_consolidation",
    "ConsolidationResult",
    "detect_manipulation",
    "ManipulationResult",
    "detect_distribution",
    "DistributionResult",
    "check_entry",
    "check_entry_with_confluence",
    "check_entry_at_candle",
    "EntrySignal",
    "calculate_risk",
    "RiskParams",
    # Entry modes
    "ENTRY_MODE_RETEST_ONLY",
    "ENTRY_MODE_RETEST_WITH_FVG",
    "ENTRY_MODE_ORDER_BLOCK",
    "ENTRY_MODE_PEAK_LOW",
    # FVG
    "FVG",
    "detect_fvg",
    "find_fvgs_in_range",
    "find_fvg_at_retest_level",
    # Order Blocks
    "OrderBlock",
    "detect_order_block",
    "find_order_blocks_in_range",
    "find_ob_at_retest_level",
    # Market Structure
    "SwingPoint",
    "StructureBreak",
    "detect_swing_high",
    "detect_swing_low",
    "find_swing_points",
    "find_bos_after_manipulation",
    "detect_break_of_structure",
]
