"""
Backtesting Engine
Main loop for simulating AMD strategy on historical data.
Supports SMC confluence features: FVG, Order Blocks, Break of Structure.
Includes realistic execution model, session/news filters, HTF bias, and more.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Set, Tuple
import pandas as pd
import numpy as np
import logging
import uuid

from config import STRATEGY, BACKTEST, VALIDATION, EXECUTION, TIME_CONFIG, SESSION_FILTER
from src.strategy.indicators import add_indicators, calculate_atr
from src.strategy.consolidation import detect_consolidation, ConsolidationResult, detect_equal_levels, score_consolidation_quality
from src.strategy.manipulation import (
    detect_manipulation,
    ManipulationResult,
    confirm_liquidity_sweep,
    confirm_volume_spike,
    score_judas_quality,
    score_judas_quality_fast,
)
from src.strategy.distribution import detect_distribution, DistributionResult, validate_distribution_strength
from src.strategy.entry import (
    check_entry, check_immediate_entry, EntrySignal,
    check_entry_at_candle, check_premium_discount_filter,
    ENTRY_MODE_RETEST_ONLY,
)
from src.strategy.risk import (
    calculate_risk,
    calculate_exit_r_multiple,
    calculate_pnl,
    calculate_pnl_with_costs,
    get_effective_stop_with_trailing,
    RiskParams,
)
from src.strategy.fvg import FVG, find_fvg_at_retest_level, find_fvgs_in_range
from src.strategy.order_blocks import OrderBlock, find_ob_at_retest_level, find_order_blocks_in_range
from src.strategy.market_structure import StructureBreak, find_bos_after_manipulation
from src.data.db import Database, Trade

# New filter imports
from src.strategy.time_filters import TimeFilterEngine, localize_dataframe_timestamps
from src.strategy.news_filter import NewsFilterEngine
from src.strategy.key_levels import KeyLevelsEngine
from src.strategy.htf_bias import HTFBiasEngine
from src.strategy.volume_filters import VolumeFilterEngine
from src.strategy.fundamentals import FundamentalsEngine
from src.backtest.execution import ExecutionEngine, FillResult, ExitDecision

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a simulated trade."""
    entry_time: datetime
    exit_time: datetime = None
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    position_size: float = 0.0
    r_multiple: float = 0.0
    pnl_pips: float = 0.0
    pnl_usd: float = 0.0
    exit_reason: str = ""  # "SL", "TP", "TIMEOUT", "ROLLOVER"

    # AMD context
    consolidation_high: float = 0.0
    consolidation_low: float = 0.0
    manipulation_extreme: float = 0.0
    manipulation_direction: str = ""

    # SMC Confluence data
    entry_mode: str = ""           # Entry mode used
    fvg_confluence: bool = False   # Had FVG at entry
    ob_confluence: bool = False    # Had Order Block at entry
    bos_confirmed: bool = False    # Break of Structure confirmed
    confluence_score: int = 0      # Number of confluence factors

    # Execution costs (new)
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    commission_cost: float = 0.0
    total_costs: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0

    # Fill info (new)
    desired_entry_price: float = 0.0
    actual_fill_price: float = 0.0
    fill_model: str = ""  # LIMIT_AT_RETEST, CLOSE, etc.

    # Filter info (new)
    htf_bias_primary: str = ""
    htf_bias_secondary: str = ""
    key_level_score: int = 0

    # Partial TP tracking
    partial_tp_taken: bool = False      # Whether partial TP was hit
    partial_tp_price: float = 0.0       # Price of partial TP (1R)
    partial_close_pct: float = 0.0      # Percentage closed at partial
    partial_pnl: float = 0.0            # PnL from partial close
    remaining_size: float = 0.0         # Size remaining after partial
    original_sl: float = 0.0            # Original SL before BE move
    sl_moved_to_be: bool = False        # Whether SL was moved to breakeven

    # Trailing stop tracking
    best_price_in_favor: float = 0.0  # Best high (LONG) or best low (SHORT) since entry
    trailing_active: bool = False     # Whether trailing stop has activated

    # Confidence sizing
    confidence_tier: str = ""        # "high", "medium", "standard", or "base"
    risk_pct_used: float = 0.0      # Actual risk % used for this trade

    # Metadata
    backtest_id: str = ""


@dataclass
class AMDPattern:
    """Tracks a complete AMD pattern."""
    consolidation: ConsolidationResult
    manipulation: ManipulationResult = None
    distribution: DistributionResult = None
    entry: EntrySignal = None
    completed: bool = False
    traded: bool = False


@dataclass
class BacktestState:
    """Current state of the backtester."""
    in_position: bool = False
    current_trade: TradeRecord = None
    position_entry_idx: int = 0

    # Track active patterns being formed
    active_patterns: List[AMDPattern] = field(default_factory=list)

    # Pattern deduplication - track seen patterns by their key features
    seen_patterns: Set[Tuple] = field(default_factory=set)


class BacktestEngine:
    """
    Engine for backtesting AMD strategy on historical data.

    Properly detects AMD patterns by:
    1. Finding consolidation zones that have ENDED (not current)
    2. Detecting manipulation after consolidation ends
    3. Confirming distribution after manipulation
    4. Entering on retest

    Includes realistic execution model, filters, and risk management.
    """

    def __init__(
        self,
        initial_capital: float = None,
        max_risk_pct: float = None,
        min_rr: float = None,
        max_trade_duration: int = None,
        # Execution options
        fill_model: str = None,
        intrabar_assumption: str = None,
        spread_points: float = None,
        slippage_model: str = None,
        commission_per_lot: float = None,
        # Filter toggles
        enable_session_filter: bool = None,
        enable_news_filter: bool = None,
        enable_htf_bias: bool = None,
        enable_key_levels: bool = None,
        enable_volume_filter: bool = None,
        enable_fundamentals: bool = None,
    ):
        self.initial_capital = initial_capital or BACKTEST["initial_capital"]
        self.max_risk_pct = max_risk_pct or STRATEGY["max_risk_pct"]
        self.min_rr = min_rr or STRATEGY["min_rr"]
        self.max_trade_duration = max_trade_duration if max_trade_duration is not None else STRATEGY.get("max_trade_duration", 200)

        self.backtest_id = str(uuid.uuid4())[:8]
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []
        self.mtm_equity_curve: List[float] = []  # Mark-to-market equity
        self.state = BacktestState()
        self.current_balance = self.initial_capital

        # Initialize execution engine
        self.execution = ExecutionEngine(
            fill_model=fill_model,
            intrabar_assumption=intrabar_assumption,
            spread_points=spread_points,
            slippage_model=slippage_model,
            commission_per_lot=commission_per_lot,
        )

        # Initialize filter engines
        self.time_filter = TimeFilterEngine(
            enabled=enable_session_filter if enable_session_filter is not None else True
        )
        self.news_filter = NewsFilterEngine(
            enabled=enable_news_filter if enable_news_filter is not None else True
        )
        self.htf_bias = HTFBiasEngine(
            enabled=enable_htf_bias if enable_htf_bias is not None else True
        )
        self.key_levels = KeyLevelsEngine(
            enabled=enable_key_levels if enable_key_levels is not None else True
        )
        self.volume_filter = VolumeFilterEngine(
            enabled=enable_volume_filter if enable_volume_filter is not None else True
        )
        self.fundamentals = FundamentalsEngine(
            enabled=enable_fundamentals if enable_fundamentals is not None else False
        )

        # Rejection tracking for funnel analysis
        self.rejection_stats = {
            "consolidations_found": 0,
            "no_manipulation": 0,
            "no_distribution": 0,
            "no_distribution_follow_through": 0,
            "no_bos": 0,
            "no_liquidity_sweep": 0,
            "no_volume_spike": 0,
            "no_entry_retest": 0,
            "entry_too_late": 0,
            "short_filtered": 0,
            "risk_invalid": 0,
            "entries_executed": 0,
            # New filter rejection counters
            "filtered_session": 0,
            "filtered_news": 0,
            "filtered_htf_bias": 0,
            "filtered_key_levels": 0,
            "filtered_volume": 0,
            "filtered_fundamentals": 0,
            "filtered_daily_limit": 0,
            "filtered_cooldown": 0,
            "filtered_rollover": 0,
            "filtered_blackout_hour": 0,
            "filtered_blackout_weekday": 0,
            "filtered_premium_discount": 0,
            "filtered_consolidation_not_asian": 0,
            "filtered_distribution_not_london_ny": 0,
            "pattern_duplicates": 0,
            "fill_not_triggered": 0,
        }

        # Confluence tracking
        self.confluence_stats = {
            "entries_with_fvg": 0,
            "entries_with_ob": 0,
            "entries_with_bos": 0,
            "avg_confluence_score": 0.0,
        }

        # Cost tracking
        self.cost_stats = {
            "total_spread_cost": 0.0,
            "total_slippage_cost": 0.0,
            "total_commission_cost": 0.0,
            "total_costs": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
        }

        # AMD conformity tracking
        self.amd_stats = {
            "consolidations_found": 0,
            "manipulations_found": 0,
            "distributions_found": 0,
            "distributions_with_bos_confluence": 0,
            "manip_to_dist_bars_total": 0,
            "manip_to_dist_count": 0,
        }

        # Minimum lookback needed
        self.lookback = STRATEGY["consolidation_lookback"]
        self.atr_period = STRATEGY["atr_period"]
        self.min_lookback = self.lookback + self.atr_period + 20

    def _build_midnight_opens(self) -> dict:
        """Pre-compute midnight (05:00 UTC) open prices keyed by date.

        Replaces the per-manipulation 300-bar backward scan with a single
        forward pass over all timestamps, yielding O(1) lookups later.
        """
        result = {}
        ts_arr = self._timestamps
        opens_arr = self._opens
        for i in range(len(ts_arr)):
            ts = pd.Timestamp(ts_arr[i])
            if hasattr(ts, "hour") and ts.hour == 5 and ts.minute == 0:
                result[ts.date()] = float(opens_arr[i])
        return result

    def run(self, df: pd.DataFrame, verbose: bool = False) -> Dict[str, Any]:
        """Run backtest on historical data."""
        if len(df) < self.min_lookback:
            logger.error(f"Not enough data. Need at least {self.min_lookback} candles.")
            return {"error": "Insufficient data"}

        # Add indicators
        df = add_indicators(
            df.copy(),
            atr_period=self.atr_period,
        )
        df = df.reset_index(drop=True)

        # Localize timestamps for session filtering
        df = localize_dataframe_timestamps(df)

        # Add HTF bias columns
        df = self.htf_bias.add_htf_bias(df)

        # Add key level columns
        df = self.key_levels.add_key_levels(df)

        # Add news blackout column
        df = self.news_filter.add_blackout_column(df)

        # Add fundamentals if enabled
        df = self.fundamentals.add_fundamentals(df)

        # Pre-extract numpy arrays for the entire dataset (avoids millions of df[col].values calls)
        self._highs = df["high"].values
        self._lows = df["low"].values
        self._opens = df["open"].values
        self._closes = df["close"].values
        self._timestamps = df["timestamp"].values
        self._atrs = df["atr"].values
        self._tick_volumes = df["tick_volume"].values if "tick_volume" in df.columns else None

        # Pre-compute rolling max/min and consolidation validity for all windows
        from numpy.lib.stride_tricks import sliding_window_view
        lookback = self.lookback
        _hw = sliding_window_view(self._highs, lookback + 1)
        _lw = sliding_window_view(self._lows, lookback + 1)
        self._roll_high_max = _hw.max(axis=1)   # shape (N - lookback,)
        self._roll_low_min = _lw.min(axis=1)

        # Pre-compute range sizes and close_pct for every window
        self._roll_range_size = self._roll_high_max - self._roll_low_min

        # Pre-compute close_pct pass/fail for all windows
        close_pct_threshold = STRATEGY["consolidation_close_pct"]
        _cw = sliding_window_view(self._closes, lookback + 1)
        # For each window, count closes inside [range_low, range_high]
        _range_low_2d = self._roll_low_min[:, np.newaxis]   # broadcast
        _range_high_2d = self._roll_high_max[:, np.newaxis]
        _inside = ((_cw >= _range_low_2d) & (_cw <= _range_high_2d)).sum(axis=1)
        self._roll_close_pct_pass = (_inside / (lookback + 1)) >= close_pct_threshold

        # Pre-compute midnight open lookup: O(1) per query instead of 300-bar backward scan
        self._midnight_opens = self._build_midnight_opens()

        # Reset state
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.mtm_equity_curve = [self.initial_capital]
        self.state = BacktestState()
        self.current_balance = self.initial_capital

        # Reset time filter session states
        self.time_filter.reset_daily_state()

        # Reset rejection stats
        self.rejection_stats = {
            "consolidations_found": 0,
            "no_manipulation": 0,
            "no_distribution": 0,
            "no_distribution_follow_through": 0,
            "no_bos": 0,
            "no_liquidity_sweep": 0,
            "no_volume_spike": 0,
            "no_entry_retest": 0,
            "entry_too_late": 0,
            "short_filtered": 0,
            "risk_invalid": 0,
            "entries_executed": 0,
            "filtered_session": 0,
            "filtered_news": 0,
            "filtered_htf_bias": 0,
            "filtered_key_levels": 0,
            "filtered_volume": 0,
            "filtered_fundamentals": 0,
            "filtered_daily_limit": 0,
            "filtered_cooldown": 0,
            "filtered_rollover": 0,
            "filtered_blackout_hour": 0,
            "filtered_blackout_weekday": 0,
            "filtered_premium_discount": 0,
            "filtered_consolidation_not_asian": 0,
            "filtered_distribution_not_london_ny": 0,
            "pattern_duplicates": 0,
            "fill_not_triggered": 0,
        }

        # Reset confluence stats
        self.confluence_stats = {
            "entries_with_fvg": 0,
            "entries_with_ob": 0,
            "entries_with_bos": 0,
            "avg_confluence_score": 0.0,
        }

        # Reset cost stats
        self.cost_stats = {
            "total_spread_cost": 0.0,
            "total_slippage_cost": 0.0,
            "total_commission_cost": 0.0,
            "total_costs": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
        }

        # Reset AMD conformity stats
        self.amd_stats = {
            "consolidations_found": 0,
            "manipulations_found": 0,
            "distributions_found": 0,
            "distributions_with_bos_confluence": 0,
            "manip_to_dist_bars_total": 0,
            "manip_to_dist_count": 0,
        }

        logger.info(f"Starting backtest on {len(df)} candles")
        logger.info(f"Date range: {self._timestamps[0]} to {self._timestamps[-1]}")

        # Main loop - process each bar
        total_bars = len(df) - self.min_lookback
        progress_interval = max(1000, total_bars // 50)  # Log every ~2%
        # Pre-compute a rolling minimum of range_size over the scan window
        # This allows us to skip bars where NO consolidation window can pass the range check
        _min_offset = STRATEGY.get("pattern_min_bars_after_consolidation", 10)
        _max_offset_cfg = STRATEGY.get("pattern_max_bars_after_consolidation", 60)
        # The scan window for bar i covers consol_start from (i - max_offset - lookback) to (i - min_offset - lookback)
        # That's a range of (max_offset - min_offset)/2 + 1 windows
        # We want min(range_size) over a sliding window of that width on self._roll_range_size
        _scan_half_width = (_max_offset_cfg - _min_offset) // 2 + 1
        if _scan_half_width > 0 and len(self._roll_range_size) > _scan_half_width:
            _rs_view = sliding_window_view(self._roll_range_size, _scan_half_width)
            self._roll_range_min_in_scan = _rs_view.min(axis=1)
        else:
            self._roll_range_min_in_scan = self._roll_range_size

        import time as _time
        _t_scan = 0.0
        _t_exit = 0.0
        _t_loop_start = _time.perf_counter()
        self._perf_consol_pass = 0
        self._perf_scan_calls = 0

        for i in range(self.min_lookback, len(df)):
            # Progress logging (so long runs don't appear stuck)
            if (i - self.min_lookback) % progress_interval == 0:
                progress_pct = ((i - self.min_lookback) / total_bars) * 100
                _elapsed = _time.perf_counter() - _t_loop_start
                logger.info(f"Processing bar {i}/{len(df)} ({progress_pct:.1f}%) - Trades: {len(self.trades)} | elapsed={_elapsed:.1f}s scan={_t_scan:.1f}s exit={_t_exit:.1f}s | scan_calls={self._perf_scan_calls} consol_pass={self._perf_consol_pass}")

            # Check open position first (df.iloc only when in position -- rare)
            if self.state.in_position:
                _t0 = _time.perf_counter()
                self._check_exit(df, i, df.iloc[i], verbose)
                _t_exit += _time.perf_counter() - _t0

            # Look for new setups if not in position
            if not self.state.in_position:
                _t0 = _time.perf_counter()
                self._scan_for_patterns(df, i, verbose)
                _t_scan += _time.perf_counter() - _t0

            # Update equity curves
            self.equity_curve.append(self.current_balance)

            # Calculate MTM equity (skip function call when not in position ~95% of bars)
            mtm_equity = self.current_balance if not self.state.in_position else self._calculate_mtm_equity(df, i)
            self.mtm_equity_curve.append(mtm_equity)

        # Close any open position at end
        if self.state.in_position:
            self._force_close(df.iloc[-1], "END_OF_DATA")

        logger.info(f"Backtest complete. Total trades: {len(self.trades)}")

        return self._generate_results()

    def _calculate_mtm_equity(self, df: pd.DataFrame, current_idx: int) -> float:
        """Calculate mark-to-market equity including unrealized P&L."""
        mtm = self.current_balance

        if self.state.in_position and self.state.current_trade:
            trade = self.state.current_trade
            current_price = float(self._closes[current_idx])

            if trade.direction == "LONG":
                unrealized = (current_price - trade.entry_price) * trade.position_size * 100
            else:
                unrealized = (trade.entry_price - current_price) * trade.position_size * 100

            mtm += unrealized

        return mtm

    def _get_pattern_key(
        self,
        consol: ConsolidationResult,
        manip: ManipulationResult,
    ) -> Tuple:
        """Create unique key for pattern deduplication."""
        return (
            consol.start_idx,
            consol.end_idx,
            round(consol.range_high, 2),
            round(consol.range_low, 2),
            manip.direction,
            manip.return_candle_idx,
        )

    def _check_filters(
        self,
        df: pd.DataFrame,
        current_idx: int,
        entry: EntrySignal,
        row: pd.Series,
        consolidation_start_idx: int = None,
        consolidation_end_idx: int = None,
        distribution_idx: int = None,
    ) -> Tuple[bool, str]:
        """
        Check all filters for entry permission.

        Returns:
            Tuple of (can_enter, rejection_reason)
        """
        ts = row.get("timestamp")
        atr = row.get("atr", 0.0)

        # 1. Session/Time filter (kill zone, Asian session, daily limits)
        can_enter, reason = self.time_filter.can_enter_trade(ts, self.initial_capital)
        if not can_enter:
            if reason == "blackout_weekday":
                self.rejection_stats["filtered_blackout_weekday"] += 1
            elif "blackout" in reason:
                self.rejection_stats["filtered_blackout_hour"] += 1
            elif "killzone" in reason or "asian" in reason:
                self.rejection_stats["filtered_session"] += 1
            elif "daily_limit" in reason:
                self.rejection_stats["filtered_daily_limit"] += 1
            elif "cooldown" in reason:
                self.rejection_stats["filtered_cooldown"] += 1
            elif "rollover" in reason:
                self.rejection_stats["filtered_rollover"] += 1
            return False, reason

        # 2. News filter
        can_enter, reason = self.news_filter.can_enter_trade(ts)
        if not can_enter:
            self.rejection_stats["filtered_news"] += 1
            return False, reason

        # 3. HTF Bias alignment
        can_enter, bias_result, reason = self.htf_bias.can_enter_trade(
            entry.direction, df, current_idx
        )
        if not can_enter:
            self.rejection_stats["filtered_htf_bias"] += 1
            return False, reason

        # 3.5 Premium/Discount zones (if require_discount_for_long / require_premium_for_short)
        if STRATEGY.get("require_discount_for_long", False) or STRATEGY.get("require_premium_for_short", False):
            entry_price = entry.desired_entry_price or entry.entry_price
            range_high = entry.consolidation_high
            range_low = entry.consolidation_low
            if range_high > range_low:
                passes, reason = check_premium_discount_filter(
                    entry_price, range_high, range_low, entry.direction
                )
                if not passes:
                    self.rejection_stats["filtered_premium_discount"] += 1
                    return False, reason or "premium_discount"

        # 4. Key levels (if mode is REQUIRE)
        can_enter, kl_score, reason = self.key_levels.can_enter_trade(
            entry.entry_price, row, atr
        )
        if not can_enter:
            self.rejection_stats["filtered_key_levels"] += 1
            return False, reason

        # 5. Volume confirmation
        if (consolidation_start_idx is not None and consolidation_end_idx is not None 
            and distribution_idx is not None):
            can_enter, volume_analysis, reason = self.volume_filter.can_enter_trade(
                df, distribution_idx, consolidation_start_idx, consolidation_end_idx
            )
            if not can_enter:
                self.rejection_stats["filtered_volume"] += 1
                return False, reason

        # 6. Fundamentals filter
        # FundamentalsEngine.can_enter_trade expects (direction, df, idx)
        can_enter, fund_result, reason = self.fundamentals.can_enter_trade(
            entry.direction, df, current_idx
        )
        if not can_enter:
            self.rejection_stats["filtered_fundamentals"] += 1
            return False, reason

        return True, ""

    def _get_confidence_tier(self, confluence_score: int, entry_hour: int) -> tuple:
        """Return (tier_name, risk_pct) based on confluence score and entry hour."""
        from config import CONFIDENCE_SIZING, RISK_MODEL

        if not CONFIDENCE_SIZING.get("enabled", False):
            return ("base", RISK_MODEL["risk_pct_per_trade_default"])

        prime_start = CONFIDENCE_SIZING.get("prime_hours_start", 13)
        prime_end = CONFIDENCE_SIZING.get("prime_hours_end", 17)
        is_prime = prime_start <= entry_hour <= prime_end

        for tier in CONFIDENCE_SIZING.get("tiers", []):
            if confluence_score >= tier["min_confluence_score"]:
                if tier.get("prime_hours_only", False) and not is_prime:
                    continue
                return (tier["name"], tier["risk_pct"])

        return ("base", CONFIDENCE_SIZING.get("base_risk_pct", 0.003))

    def _scan_for_patterns(self, df: pd.DataFrame, current_idx: int, verbose: bool):
        """
        Scan for complete AMD patterns looking backwards.

        Strategy:
        1. Look for consolidation that ended 5-30 bars ago
        2. Check if manipulation occurred after it
        3. Check if distribution confirmed
        4. Check if current price offers retest entry
        """
        lookback = self.lookback
        atr = self._atrs[current_idx]

        self._perf_scan_calls += 1

        if pd.isna(atr) or atr == 0:
            return

        highs = self._highs
        lows = self._lows
        closes = self._closes

        # Cache config lookups as locals
        max_range_mult = STRATEGY["consolidation_range_atr_mult"]
        min_offset = STRATEGY.get("pattern_min_bars_after_consolidation", 10)
        max_offset_cfg = STRATEGY.get("pattern_max_bars_after_consolidation", 60)

        # Pre-computed rolling arrays (local refs for speed)
        roll_high_max = self._roll_high_max
        roll_low_min = self._roll_low_min
        roll_range_size = self._roll_range_size
        max_range = max_range_mult * atr
        atr_period = self.atr_period

        max_offset = min(max_offset_cfg, current_idx - lookback)
        if max_offset <= min_offset:
            return

        # Quick check: can ANY window in scan range pass the range check?
        # Use pre-computed rolling minimum of range_size
        hi_start = current_idx - min_offset - lookback
        lo_start = current_idx - max_offset - lookback
        lo_start = max(lo_start, atr_period)
        if hi_start < atr_period:
            return

        range_min_arr = self._roll_range_min_in_scan
        # The rolling min array is indexed by the start of the min-window
        # We need min(range_size[lo_start:hi_start+1])
        # range_min_arr[k] = min(range_size[k:k+scan_half_width])
        if lo_start < len(range_min_arr):
            if range_min_arr[lo_start] > max_range:
                return

        # Scan backwards for consolidation zones (step by 2 for speed)
        for consol_end_offset in range(min_offset, max_offset, 2):
            consol_end_idx = current_idx - consol_end_offset
            consol_start_idx = consol_end_idx - lookback

            if consol_start_idx < atr_period:
                continue

            # Fast consolidation check: two scalar comparisons
            if roll_range_size[consol_start_idx] > max_range:
                continue

            range_high = roll_high_max[consol_start_idx]
            range_low = roll_low_min[consol_start_idx]

            self._perf_consol_pass += 1
            self.rejection_stats["consolidations_found"] += 1
            self.amd_stats["consolidations_found"] += 1

            # Create consolidation result (lightweight, before expensive checks)
            consol = ConsolidationResult(
                valid=True,
                range_high=range_high,
                range_low=range_low,
                range_size=range_high - range_low,
                atr=atr,
                start_idx=consol_start_idx,
                end_idx=consol_end_idx,
            )

            # Check manipulation FIRST (cheap numpy pre-check eliminates ~80%)
            manip = self._check_manipulation_after(df, consol, current_idx)
            if not manip.valid:
                self.rejection_stats["no_manipulation"] += 1
                continue
            self.amd_stats["manipulations_found"] += 1

            # Only enrich with equal levels AFTER manipulation passes
            if STRATEGY.get("detect_equal_levels", True):
                consol = detect_equal_levels(df, consol, highs_arr=self._highs, lows_arr=self._lows)

            # Quality gate: reject low-quality consolidations
            min_consol_quality = STRATEGY.get("min_consolidation_quality", 0)
            if min_consol_quality > 0:
                quality = score_consolidation_quality(consol)
                if quality < min_consol_quality:
                    self.rejection_stats.setdefault("low_consolidation_quality", 0)
                    self.rejection_stats["low_consolidation_quality"] += 1
                    continue

            # Score Judas swing quality (candle count, velocity, session)
            manip = score_judas_quality_fast(self._timestamps, self._midnight_opens, manip)

            # Hard gate: reject low-quality Judas swings
            min_judas = STRATEGY.get("min_judas_quality", 0)
            if min_judas > 0 and getattr(manip, "judas_quality", 0) < min_judas:
                self.rejection_stats.setdefault("no_judas_quality", 0)
                self.rejection_stats["no_judas_quality"] += 1
                continue

            # Confirm liquidity sweep if required
            if STRATEGY.get("require_liquidity_sweep", False):
                manip = confirm_liquidity_sweep(df, manip)
                if not manip.swept_liquidity:
                    self.rejection_stats["no_liquidity_sweep"] += 1
                    continue

            # Confirm volume spike during manipulation if required
            manip = confirm_volume_spike(df, manip)
            if STRATEGY.get("require_manipulation_volume_spike", False) and not getattr(manip, "volume_confirmed", False):
                self.rejection_stats["no_volume_spike"] += 1
                continue

            # Pattern deduplication - skip if we've seen this exact pattern
            pattern_key = self._get_pattern_key(consol, manip)
            if pattern_key in self.state.seen_patterns:
                self.rejection_stats["pattern_duplicates"] += 1
                continue

            # Check for distribution AFTER manipulation
            dist = self._check_distribution_after(df, consol, manip, current_idx)
            if not dist.valid:
                self.rejection_stats["no_distribution"] += 1
                continue
            follow_through_candles = STRATEGY.get("distribution_follow_through_candles", 2)
            if not validate_distribution_strength(df, dist, min_follow_through_candles=follow_through_candles):
                self.rejection_stats["no_distribution_follow_through"] += 1
                continue

            self.amd_stats["distributions_found"] += 1
            bars_to_distribution = dist.break_candle_idx - manip.return_candle_idx
            if bars_to_distribution >= 0:
                self.amd_stats["manip_to_dist_bars_total"] += bars_to_distribution
                self.amd_stats["manip_to_dist_count"] += 1

            # Session timing: Asian accumulation -> London/NY distribution
            if SESSION_FILTER.get("require_consolidation_in_asian", False):
                consol_start_ts = pd.Timestamp(self._timestamps[consol.start_idx])
                consol_end_ts = pd.Timestamp(self._timestamps[consol.end_idx])
                if not self.time_filter.consolidation_formed_in_asian(consol_start_ts, consol_end_ts):
                    self.rejection_stats["filtered_consolidation_not_asian"] += 1
                    continue
            if SESSION_FILTER.get("require_distribution_in_london_ny", False):
                dist_ts = pd.Timestamp(self._timestamps[dist.break_candle_idx])
                if not self.time_filter.distribution_in_london_ny(dist_ts):
                    self.rejection_stats["filtered_distribution_not_london_ny"] += 1
                    continue

            # Skip stale retests (too many bars after distribution)
            max_bars_after_dist = STRATEGY.get("max_bars_after_distribution")
            if max_bars_after_dist is not None:
                bars_since_dist = current_idx - dist.break_candle_idx
                if bars_since_dist > max_bars_after_dist:
                    self.rejection_stats["entry_too_late"] += 1
                    continue

            # Detect Break of Structure (optional)
            bos = None
            if STRATEGY.get("bos_required", False) or STRATEGY.get("entry_mode", "") != ENTRY_MODE_RETEST_ONLY:
                expected_bos_dir = "BULLISH" if manip.direction == "DOWN" else "BEARISH"
                bos = find_bos_after_manipulation(
                    df, manip.return_candle_idx, expected_bos_dir,
                    search_window=20, swing_lookback=STRATEGY.get("bos_swing_lookback", 5),
                    highs_arr=self._highs, lows_arr=self._lows, closes_arr=self._closes,
                )

                if STRATEGY.get("bos_required", False) and (bos is None or not bos.valid):
                    self.rejection_stats["no_bos"] += 1
                    continue

            # Detect FVG and Order Blocks for confluence
            # Always search for FVG/OB to build confluence score
            fvg_at_level = None
            ob_at_level = None
            entry_mode = STRATEGY.get("entry_mode", ENTRY_MODE_RETEST_ONLY)

            if dist.direction == "UP":
                retest_level = consol.range_high
                fvg_direction = "BULLISH"
                ob_direction = "BULLISH"
            else:
                retest_level = consol.range_low
                fvg_direction = "BEARISH"
                ob_direction = "BEARISH"

            # Find FVG near retest level (needed for confluence scoring)
            fvg_at_level = find_fvg_at_retest_level(
                df, retest_level,
                search_start_idx=consol.start_idx,
                search_end_idx=min(dist.break_candle_idx + 10, current_idx),
                direction=fvg_direction,
                atr=atr,
                tolerance_mult=STRATEGY.get("retest_tolerance_atr_mult", 0.4)
            )

            # Find Order Block near retest level (needed for confluence scoring)
            ob_at_level = find_ob_at_retest_level(
                df, retest_level,
                search_start_idx=consol.start_idx,
                search_end_idx=min(dist.break_candle_idx + 10, current_idx),
                direction=ob_direction,
                atr=atr,
                tolerance_mult=STRATEGY.get("retest_tolerance_atr_mult", 0.4)
            )

            # Track distributions with BOS + confluence (AMD conformity)
            equal_level_swept = False
            if manip.direction == "UP":
                equal_level_swept = bool(
                    consol.has_equal_highs
                    and consol.equal_high_level > 0
                    and manip.extreme_price >= consol.equal_high_level - 1e-9
                )
            elif manip.direction == "DOWN":
                equal_level_swept = bool(
                    consol.has_equal_lows
                    and consol.equal_low_level > 0
                    and manip.extreme_price <= consol.equal_low_level + 1e-9
                )

            volume_confirmed = getattr(manip, "volume_confirmed", False)
            judas_quality = getattr(manip, "judas_quality", 0)
            confluence_score = 0
            if bos is not None and bos.valid:
                confluence_score += 1
            if fvg_at_level:
                confluence_score += 1
            if ob_at_level:
                confluence_score += 1
            if equal_level_swept:
                confluence_score += 1
            if volume_confirmed:
                confluence_score += 1
            if judas_quality >= 2:
                confluence_score += 1

            min_confluence = STRATEGY.get("min_confluence_score", 0)
            if (bos is not None and bos.valid) and confluence_score >= min_confluence:
                self.amd_stats["distributions_with_bos_confluence"] += 1

            # Check for entry at current bar with confluence
            entry = self._check_entry_now_with_confluence(
                df, consol, manip, dist, current_idx,
                bos, fvg_at_level, ob_at_level
            )
            if not entry.valid:
                self.rejection_stats["no_entry_retest"] += 1
                continue

            # Check directional filter (skip shorts if disabled)
            if entry.direction == "SHORT" and not STRATEGY.get("allow_short_trades", True):
                self.rejection_stats["short_filtered"] += 1
                continue

            # Deferred row creation -- only when entry is valid and filters need a Series
            row = df.iloc[current_idx]

            # Check all filters
            can_enter, filter_reason = self._check_filters(
                df, current_idx, entry, row, 
                consol.start_idx, consol.end_idx, dist.break_candle_idx
            )
            if not can_enter:
                if verbose:
                    logger.debug(f"Entry filtered: {filter_reason}")
                continue

            # Get confidence tier for position sizing
            entry_hour = row.get("timestamp").hour if hasattr(row.get("timestamp"), "hour") else 0
            confluence_score = getattr(entry, "confluence_score", 0)
            confidence_tier, tier_risk_pct = self._get_confidence_tier(confluence_score, entry_hour)

            # Calculate risk with confidence-based sizing
            risk = calculate_risk(entry, self.current_balance, self.min_rr, self.max_risk_pct,
                                  risk_pct_override=tier_risk_pct)
            if not risk.valid:
                self.rejection_stats["risk_invalid"] += 1
                if verbose and risk.rejection_reason:
                    logger.debug(f"Risk rejected: {risk.rejection_reason}")
                continue

            # Simulate execution fill
            fill_result = self.execution.simulate_entry_fill(
                entry=entry,
                candle=row,
                atr=atr,
                position_size=risk.position_size,
            )

            if not fill_result.filled:
                self.rejection_stats["fill_not_triggered"] += 1
                if verbose:
                    logger.debug(f"Fill not triggered: {fill_result.fill_reason}")
                continue

            # Mark pattern as seen
            self.state.seen_patterns.add(pattern_key)

            # Track confluence stats
            if entry.fvg_confluence:
                self.confluence_stats["entries_with_fvg"] += 1
            if entry.ob_confluence:
                self.confluence_stats["entries_with_ob"] += 1
            if entry.bos_confirmed:
                self.confluence_stats["entries_with_bos"] += 1

            # Execute entry with filled price
            self.rejection_stats["entries_executed"] += 1
            self._pending_confidence_tier = confidence_tier
            self._pending_risk_pct = tier_risk_pct
            self._execute_entry(entry, risk, fill_result, row, current_idx, df, verbose)

            # Record trade for session tracking
            ts = row.get("timestamp")
            self.time_filter.record_trade(ts)

            return  # Only one entry per bar

    def _is_consolidation(self, window: pd.DataFrame, atr: float) -> bool:
        """Check if window forms a valid consolidation."""
        range_high = window['high'].max()
        range_low = window['low'].min()
        range_size = range_high - range_low

        # Range must be tight
        max_range = STRATEGY["consolidation_range_atr_mult"] * atr
        if range_size > max_range:
            return False

        # Check closes inside range
        closes = window['close']
        closes_inside = ((closes >= range_low) & (closes <= range_high)).sum()
        close_pct = closes_inside / len(window)

        if close_pct < STRATEGY["consolidation_close_pct"]:
            return False

        return True

    def _is_consolidation_arrays(self, high_arr: np.ndarray, low_arr: np.ndarray, close_arr: np.ndarray, atr: float) -> bool:
        """Check consolidation using numpy arrays (faster hot path)."""
        range_high = np.max(high_arr)
        range_low = np.min(low_arr)
        range_size = range_high - range_low

        max_range = STRATEGY["consolidation_range_atr_mult"] * atr
        if range_size > max_range:
            return False

        closes_inside = np.sum((close_arr >= range_low) & (close_arr <= range_high))
        close_pct = closes_inside / len(close_arr)

        if close_pct < STRATEGY["consolidation_close_pct"]:
            return False

        return True

    def _check_manipulation_after(
        self,
        df: pd.DataFrame,
        consol: ConsolidationResult,
        current_idx: int,
    ) -> ManipulationResult:
        """Check for manipulation that occurred after consolidation."""

        range_high = consol.range_high
        range_low = consol.range_low
        atr = consol.atr
        min_break = STRATEGY["manipulation_break_atr_mult"] * atr
        max_return_candles = STRATEGY["manipulation_return_candles"]

        # Search window: from consolidation end to current
        search_start = consol.end_idx + 1
        search_end = min(current_idx, search_start + 30)  # Limit search

        if search_start >= search_end:
            return ManipulationResult(valid=False)

        # Quick numpy pre-check: can any fakeout exist in this window?
        h_slice = self._highs[search_start:search_end]
        l_slice = self._lows[search_start:search_end]
        has_up_break = h_slice.max() > range_high + min_break
        has_down_break = l_slice.min() < range_low - min_break

        if not has_up_break and not has_down_break:
            return ManipulationResult(valid=False)

        # Check for upward fakeout
        if has_up_break:
            upward = self._find_fakeout(
                search_start, search_end,
                range_high, range_low, min_break, max_return_candles,
                direction="UP", atr=atr
            )
            if upward.valid:
                return upward

        # Check for downward fakeout
        if has_down_break:
            downward = self._find_fakeout(
                search_start, search_end,
                range_high, range_low, min_break, max_return_candles,
                direction="DOWN", atr=atr
            )
            return downward

        return ManipulationResult(valid=False)

    def _find_fakeout(
        self,
        start_idx: int,
        end_idx: int,
        range_high: float,
        range_low: float,
        min_break: float,
        max_return_candles: int,
        direction: str,
        atr: float,
    ) -> ManipulationResult:
        """Find a fakeout pattern in the given range (uses pre-extracted arrays)."""
        h = self._highs
        l_ = self._lows
        c = self._closes

        break_idx = -1
        extreme = 0.0 if direction == "UP" else float("inf")

        for i in range(start_idx, end_idx):
            hi = h[i]
            lo = l_[i]
            cl = c[i]

            if direction == "UP":
                if hi > range_high + min_break:
                    if break_idx == -1:
                        break_idx = i
                    extreme = max(extreme, hi)

                if break_idx >= 0:
                    if cl <= range_high and cl >= range_low:
                        if i - break_idx <= max_return_candles:
                            return ManipulationResult(
                                valid=True,
                                direction="UP",
                                extreme_price=extreme,
                                break_distance=extreme - range_high,
                                return_candle_idx=i,
                                atr=atr,
                                manipulation_candle_count=max(1, i - break_idx),
                            )
                    if i - break_idx > max_return_candles:
                        break_idx = -1
                        extreme = 0.0
            else:
                if lo < range_low - min_break:
                    if break_idx == -1:
                        break_idx = i
                    extreme = min(extreme, lo)

                if break_idx >= 0:
                    if cl >= range_low and cl <= range_high:
                        if i - break_idx <= max_return_candles:
                            return ManipulationResult(
                                valid=True,
                                direction="DOWN",
                                extreme_price=extreme,
                                break_distance=range_low - extreme,
                                return_candle_idx=i,
                                atr=atr,
                                manipulation_candle_count=max(1, i - break_idx),
                            )
                    if i - break_idx > max_return_candles:
                        break_idx = -1
                        extreme = float("inf")

        return ManipulationResult(valid=False)

    def _check_distribution_after(
        self,
        df: pd.DataFrame,
        consol: ConsolidationResult,
        manip: ManipulationResult,
        current_idx: int,
    ) -> DistributionResult:
        """Check for distribution after manipulation (uses pre-extracted arrays)."""
        o = self._opens
        c = self._closes

        atr = manip.atr
        min_break = STRATEGY["distribution_break_atr_mult"] * atr
        body_mult = STRATEGY["distribution_body_mult"]

        range_high = consol.range_high
        range_low = consol.range_low

        expected_dir = "UP" if manip.direction == "DOWN" else "DOWN"

        # Average body size from consolidation (numpy slice)
        s, e = consol.start_idx, consol.end_idx + 1
        avg_body = float(np.mean(np.abs(c[s:e] - o[s:e])))
        if avg_body == 0:
            avg_body = atr * 0.1

        search_start = manip.return_candle_idx + 1
        search_end = min(current_idx, search_start + 20)

        for i in range(search_start, search_end):
            close_i = c[i]
            body = abs(close_i - o[i])
            body_ratio = body / avg_body if avg_body > 0 else 0

            if expected_dir == "UP":
                break_distance = close_i - range_high
                if break_distance >= min_break and body_ratio >= body_mult:
                    return DistributionResult(
                        valid=True,
                        direction="UP",
                        break_price=close_i,
                        break_distance=break_distance,
                        body_expansion=body_ratio,
                        break_candle_idx=i,
                        atr=atr,
                    )
            else:
                break_distance = range_low - close_i
                if break_distance >= min_break and body_ratio >= body_mult:
                    return DistributionResult(
                        valid=True,
                        direction="DOWN",
                        break_price=close_i,
                        break_distance=break_distance,
                        body_expansion=body_ratio,
                        break_candle_idx=i,
                        atr=atr,
                    )

        return DistributionResult(valid=False)

    def _check_entry_now(
        self,
        df: pd.DataFrame,
        consol: ConsolidationResult,
        manip: ManipulationResult,
        dist: DistributionResult,
        current_idx: int,
    ) -> EntrySignal:
        """Check if current bar provides valid entry."""

        candle = df.iloc[current_idx]
        atr = dist.atr
        tolerance = STRATEGY["retest_tolerance_atr_mult"] * atr
        wick_ratio = STRATEGY["rejection_wick_ratio"]

        if dist.direction == "UP":
            # Long setup - retest of range_high from above
            retest_level = consol.range_high
            direction = "LONG"

            # Check if low retests the level
            if candle["low"] <= retest_level + tolerance and candle["low"] >= retest_level - tolerance:
                # Check for rejection (bullish)
                body = abs(candle["close"] - candle["open"])
                lower_wick = min(candle["open"], candle["close"]) - candle["low"]

                if body == 0 or lower_wick >= body * wick_ratio:
                    return EntrySignal(
                        valid=True,
                        direction=direction,
                        entry_price=candle["close"],
                        entry_candle_idx=current_idx,
                        entry_timestamp=candle.get("timestamp"),
                        rejection_confirmed=True,
                        retest_level=retest_level,
                        consolidation_high=consol.range_high,
                        consolidation_low=consol.range_low,
                        manipulation_extreme=manip.extreme_price,
                        manipulation_direction=manip.direction,
                    )
        else:
            # Short setup - retest of range_low from below
            retest_level = consol.range_low
            direction = "SHORT"

            # Check if high retests the level
            if candle["high"] >= retest_level - tolerance and candle["high"] <= retest_level + tolerance:
                # Check for rejection (bearish)
                body = abs(candle["close"] - candle["open"])
                upper_wick = candle["high"] - max(candle["open"], candle["close"])

                if body == 0 or upper_wick >= body * wick_ratio:
                    return EntrySignal(
                        valid=True,
                        direction=direction,
                        entry_price=candle["close"],
                        entry_candle_idx=current_idx,
                        entry_timestamp=candle.get("timestamp"),
                        rejection_confirmed=True,
                        retest_level=retest_level,
                        consolidation_high=consol.range_high,
                        consolidation_low=consol.range_low,
                        manipulation_extreme=manip.extreme_price,
                        manipulation_direction=manip.direction,
                    )

        return EntrySignal(valid=False)

    def _check_entry_now_with_confluence(
        self,
        df: pd.DataFrame,
        consol: ConsolidationResult,
        manip: ManipulationResult,
        dist: DistributionResult,
        current_idx: int,
        bos: StructureBreak = None,
        fvg_at_level: FVG = None,
        ob_at_level: OrderBlock = None,
    ) -> EntrySignal:
        """Check if current bar provides valid entry with SMC confluence."""

        entry_mode = STRATEGY.get("entry_mode", ENTRY_MODE_RETEST_ONLY)

        # Use the new confluence-aware entry check
        entry = check_entry_at_candle(
            df=df,
            current_idx=current_idx,
            consolidation=consol,
            manipulation=manip,
            distribution=dist,
            structure_break=bos,
            fvg_at_level=fvg_at_level,
            ob_at_level=ob_at_level,
            entry_mode=entry_mode,
        )

        return entry

    def _execute_entry(
        self,
        entry: EntrySignal,
        risk: RiskParams,
        fill: FillResult,
        current_candle: pd.Series,
        current_idx: int,
        df: pd.DataFrame,
        verbose: bool,
    ):
        """Execute trade entry with execution costs."""

        # Get HTF bias info for record
        htf_primary = current_candle.get("htf_bias_primary", "")
        htf_secondary = current_candle.get("htf_bias_secondary", "")

        # Get key level score
        atr = current_candle.get("atr", 0.0)
        kl_score = self.key_levels.calculate_score(fill.fill_price, current_candle, atr)

        best_price = fill.fill_price  # Initialize for trailing stop
        trade = TradeRecord(
            entry_time=current_candle.get("timestamp", datetime.now()),
            direction=entry.direction,
            entry_price=fill.fill_price,  # Use actual fill price
            sl_price=risk.stop_loss,
            tp_price=risk.take_profit,
            original_sl=risk.stop_loss,
            sl_moved_to_be=False,
            best_price_in_favor=best_price,
            position_size=risk.position_size,
            consolidation_high=entry.consolidation_high,
            consolidation_low=entry.consolidation_low,
            manipulation_extreme=entry.manipulation_extreme,
            manipulation_direction=entry.manipulation_direction,
            entry_mode=getattr(entry, 'entry_mode', ''),
            fvg_confluence=getattr(entry, 'fvg_confluence', False),
            ob_confluence=getattr(entry, 'ob_confluence', False),
            bos_confirmed=getattr(entry, 'bos_confirmed', False),
            confluence_score=getattr(entry, 'confluence_score', 0),
            # Execution costs
            spread_cost=fill.costs.spread_cost,
            slippage_cost=fill.costs.slippage_cost,
            commission_cost=fill.costs.commission_cost,
            total_costs=fill.costs.total_cost,
            # Fill info
            desired_entry_price=entry.desired_entry_price,
            actual_fill_price=fill.fill_price,
            fill_model=fill.fill_model,
            # Filter info
            htf_bias_primary=htf_primary,
            htf_bias_secondary=htf_secondary,
            key_level_score=kl_score.total_score,
            backtest_id=self.backtest_id,
            confidence_tier=getattr(self, '_pending_confidence_tier', 'base'),
            risk_pct_used=getattr(self, '_pending_risk_pct', 0.003),
        )

        self.state.in_position = True
        self.state.current_trade = trade
        self.state.position_entry_idx = current_idx

        if verbose:
            logger.info(
                f"ENTRY: {entry.direction} @ {fill.fill_price:.2f} | "
                f"SL: {risk.stop_loss:.2f} | TP: {risk.take_profit:.2f} | "
                f"Size: {risk.position_size:.2f} | "
                f"Costs: ${fill.costs.total_cost:.2f} | "
                f"Tier: {trade.confidence_tier} ({trade.risk_pct_used*100:.1f}%)"
            )

    def _check_exit(self, df: pd.DataFrame, current_idx: int, candle: pd.Series, verbose: bool):
        """Check if current position should be exited (supports partial TP)."""
        from config import RISK_MODEL

        trade = self.state.current_trade
        ts = candle.get("timestamp")
        atr = candle.get("atr", 0.0)

        # Check timeout
        bars_in_trade = current_idx - self.state.position_entry_idx
        if bars_in_trade >= self.max_trade_duration:
            self._exit_position(candle["close"], "TIMEOUT", candle, verbose)
            return

        # Check rollover exit
        if self.time_filter.should_close_for_rollover(ts):
            self._exit_position(candle["close"], "ROLLOVER", candle, verbose)
            return

        # Update best price in favor (for trailing stop)
        if trade.direction == "LONG":
            trade.best_price_in_favor = max(
                getattr(trade, "best_price_in_favor", trade.entry_price),
                candle["high"],
            )
        else:
            trade.best_price_in_favor = min(
                getattr(trade, "best_price_in_favor", trade.entry_price),
                candle["low"],
            )

        # Move SL to breakeven at configured R level (before trailing stop)
        be_trigger_r = STRATEGY.get("move_sl_to_be_at_r", 0)
        if be_trigger_r > 0 and atr and atr > 0 and not getattr(trade, "sl_moved_to_be", False):
            sl_for_be = trade.original_sl if trade.original_sl > 0 else trade.sl_price
            stop_distance_be = (
                trade.entry_price - sl_for_be
                if trade.direction == "LONG"
                else sl_for_be - trade.entry_price
            )
            if stop_distance_be > 0:
                if trade.direction == "LONG":
                    current_r = (trade.best_price_in_favor - trade.entry_price) / stop_distance_be
                else:
                    current_r = (trade.entry_price - trade.best_price_in_favor) / stop_distance_be

                if current_r >= be_trigger_r:
                    be_buffer = atr * 0.1
                    if trade.direction == "LONG":
                        new_be_sl = trade.entry_price + be_buffer
                        if new_be_sl > trade.sl_price:
                            trade.sl_price = new_be_sl
                    else:
                        new_be_sl = trade.entry_price - be_buffer
                        if new_be_sl < trade.sl_price:
                            trade.sl_price = new_be_sl
                    trade.sl_moved_to_be = True

        # Apply trailing stop if enabled
        if STRATEGY.get("trailing_stop_enabled", False) and atr and atr > 0:
            # Use original SL for R calculation so trailing works after BE move
            ref_sl = trade.original_sl if trade.original_sl > 0 else trade.sl_price
            stop_distance = (
                trade.entry_price - ref_sl
                if trade.direction == "LONG"
                else ref_sl - trade.entry_price
            )
            if stop_distance > 0:
                effective_sl, trailing_active = get_effective_stop_with_trailing(
                    entry_price=trade.entry_price,
                    current_sl=trade.sl_price,
                    current_extreme_price=trade.best_price_in_favor,
                    stop_distance=stop_distance,
                    atr=atr,
                    direction=trade.direction,
                )
                if trailing_active and effective_sl != trade.sl_price:
                    trade.sl_price = effective_sl
                if trailing_active:
                    trade.trailing_active = True

        # Disable TP when trailing is active — let the trail manage the exit
        if STRATEGY.get("disable_tp_when_trailing", False) and trade.trailing_active:
            if trade.direction == "LONG":
                trade.tp_price = float('inf')
            else:
                trade.tp_price = 0.0

        # Use execution engine for exit decision (with partial TP support)
        if RISK_MODEL.get("partial_tp_enabled", False):
            exit_decision = self.execution.check_exit_with_partial(
                trade=trade,
                candle=candle,
                atr=atr,
            )
        else:
            exit_decision = self.execution.check_exit(
                trade=trade,
                candle=candle,
                atr=atr,
            )

        # Handle partial close
        if exit_decision.is_partial:
            self._handle_partial_close(exit_decision, candle, verbose)
            return

        if exit_decision.should_exit:
            self._exit_position(exit_decision.exit_price, exit_decision.exit_reason, candle, verbose)
            return

    def _handle_partial_close(self, exit_decision: ExitDecision, candle: pd.Series, verbose: bool):
        """Handle partial TP close and SL move to breakeven."""
        from config import RISK_MODEL

        trade = self.state.current_trade
        contract_size = RISK_MODEL.get("contract_size", 100)

        # Store original SL if not already stored
        if trade.original_sl == 0:
            trade.original_sl = trade.sl_price

        # Calculate partial PnL
        partial_size = trade.position_size * exit_decision.partial_close_pct
        if trade.direction == "LONG":
            partial_pnl = (exit_decision.exit_price - trade.entry_price) * partial_size * contract_size
        else:
            partial_pnl = (trade.entry_price - exit_decision.exit_price) * partial_size * contract_size

        # Update trade record
        trade.partial_tp_taken = True
        trade.partial_tp_price = exit_decision.exit_price
        trade.partial_close_pct = exit_decision.partial_close_pct
        trade.partial_pnl = partial_pnl
        trade.remaining_size = trade.position_size * (1 - exit_decision.partial_close_pct)

        # Move SL to breakeven if configured
        if exit_decision.new_sl_price > 0:
            trade.sl_price = exit_decision.new_sl_price
            trade.sl_moved_to_be = True

        # Credit partial PnL to balance
        self.current_balance += partial_pnl

        if verbose:
            logger.info(
                f"PARTIAL TP: {trade.direction} closed {exit_decision.partial_close_pct*100:.0f}% "
                f"@ {exit_decision.exit_price:.2f} | PnL: ${partial_pnl:.2f} | "
                f"New SL: {trade.sl_price:.2f} (BE)"
            )

    def _exit_position(self, exit_price: float, reason: str, candle: pd.Series, verbose: bool):
        """Exit current position with cost calculation (accounts for partial TP)."""
        trade = self.state.current_trade
        atr = candle.get("atr", 0.0)

        trade.exit_price = exit_price
        trade.exit_time = candle.get("timestamp", datetime.now())
        trade.exit_reason = reason

        # Use original SL for R-multiple calculation if SL was moved to BE
        sl_for_r = trade.original_sl if trade.original_sl > 0 else trade.sl_price

        # Calculate P&L with costs using contract-size math
        trade.r_multiple = calculate_exit_r_multiple(
            trade.entry_price, exit_price, sl_for_r, trade.direction
        )

        # Use remaining size if partial TP was taken
        position_size_for_exit = trade.remaining_size if trade.partial_tp_taken else trade.position_size

        # Calculate gross and net P&L (with costs)
        gross_pnl, net_pnl, total_costs = calculate_pnl_with_costs(
            entry_price=trade.entry_price,
            exit_price=exit_price,
            position_size=position_size_for_exit,
            direction=trade.direction,
            commission=trade.commission_cost * (1 - trade.partial_close_pct) if trade.partial_tp_taken else trade.commission_cost,
            spread_cost=trade.spread_cost * (1 - trade.partial_close_pct) if trade.partial_tp_taken else trade.spread_cost,
            slippage_cost=trade.slippage_cost * (1 - trade.partial_close_pct) if trade.partial_tp_taken else trade.slippage_cost,
            swap_cost=0.0,  # Swap costs handled separately if needed
        )

        # Add partial PnL to gross/net if partial was taken
        if trade.partial_tp_taken:
            gross_pnl += trade.partial_pnl
            net_pnl += trade.partial_pnl

        trade.gross_pnl = gross_pnl
        trade.net_pnl = net_pnl
        trade.pnl_usd = net_pnl  # For backwards compatibility
        trade.total_costs = total_costs  # Update total costs from calculation

        # Update balance with NET P&L (partial already credited, so only add remainder)
        if trade.partial_tp_taken:
            self.current_balance += (net_pnl - trade.partial_pnl)  # Only add the exit portion
        else:
            self.current_balance += net_pnl

        # Update cost stats
        self.cost_stats["total_spread_cost"] += trade.spread_cost
        self.cost_stats["total_slippage_cost"] += trade.slippage_cost
        self.cost_stats["total_commission_cost"] += trade.commission_cost
        self.cost_stats["total_costs"] += trade.total_costs
        self.cost_stats["gross_pnl"] += gross_pnl
        self.cost_stats["net_pnl"] += net_pnl

        # Record exit PnL for session tracking
        ts = candle.get("timestamp")
        self.time_filter.record_trade_exit(ts, net_pnl)

        # Save trade
        self.trades.append(trade)

        # Reset state
        self.state.in_position = False
        self.state.current_trade = None

        if verbose:
            result = "WIN" if net_pnl > 0 else "LOSS"
            logger.info(
                f"EXIT ({result}): {trade.direction} @ {exit_price:.2f} | "
                f"Reason: {reason} | R: {trade.r_multiple:.2f} | "
                f"Gross: ${gross_pnl:.2f} | Net: ${net_pnl:.2f}"
            )

    def _force_close(self, candle: pd.Series, reason: str):
        """Force close position at end of data."""
        if self.state.in_position:
            self._exit_position(candle["close"], reason, candle, verbose=True)

    def _generate_results(self) -> Dict[str, Any]:
        """Generate backtest results summary."""
        if not self.trades:
            amd_conformity = {
                "consolidations_found": self.amd_stats.get("consolidations_found", 0),
                "manipulations_found": self.amd_stats.get("manipulations_found", 0),
                "distributions_found": self.amd_stats.get("distributions_found", 0),
                "distributions_with_bos_confluence": self.amd_stats.get("distributions_with_bos_confluence", 0),
                "consolidation_to_manipulation_pct": 0.0,
                "manipulation_to_distribution_pct": 0.0,
                "distribution_with_bos_confluence_pct": 0.0,
                "avg_bars_manipulation_to_distribution": 0.0,
            }
            return {
                "backtest_id": self.backtest_id,
                "total_trades": 0,
                "error": "No trades generated",
                "funnel_stats": self.rejection_stats,
                "amd_conformity": amd_conformity,
            }

        # Convert to DataFrame for analysis
        trades_df = pd.DataFrame([vars(t) for t in self.trades])

        # Basic stats
        total_trades = len(self.trades)
        wins = len(trades_df[trades_df["net_pnl"] > 0])
        losses = len(trades_df[trades_df["net_pnl"] <= 0])
        win_rate = wins / total_trades if total_trades > 0 else 0

        # R-multiple stats
        avg_r = trades_df["r_multiple"].mean()
        avg_win_r = trades_df[trades_df["r_multiple"] > 0]["r_multiple"].mean() if wins > 0 else 0
        avg_loss_r = trades_df[trades_df["r_multiple"] <= 0]["r_multiple"].mean() if losses > 0 else 0

        # Expectancy
        expectancy = avg_r

        # P&L - use net_pnl (after costs)
        total_net_pnl = trades_df["net_pnl"].sum()
        total_gross_pnl = trades_df["gross_pnl"].sum()
        gross_profit = trades_df[trades_df["net_pnl"] > 0]["net_pnl"].sum()
        gross_loss = abs(trades_df[trades_df["net_pnl"] <= 0]["net_pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Drawdown - use MTM equity curve
        equity = pd.Series(self.mtm_equity_curve)
        rolling_max = equity.cummax()
        drawdown = (rolling_max - equity) / rolling_max
        max_drawdown = drawdown.max()

        # By exit reason
        exit_reason_counts = trades_df["exit_reason"].value_counts().to_dict()
        sl_exits = exit_reason_counts.get("SL", 0)
        tp_exits = exit_reason_counts.get("TP", 0)
        rollover_exits = exit_reason_counts.get("ROLLOVER", 0)
        timeout_exits = exit_reason_counts.get("TIMEOUT", 0)

        # Calculate average confluence score
        if total_trades > 0:
            if "confluence_score" in trades_df.columns:
                self.confluence_stats["avg_confluence_score"] = round(
                    trades_df["confluence_score"].mean(), 2
                )

        # AMD conformity summary
        consol_count = self.amd_stats.get("consolidations_found", 0)
        manip_count = self.amd_stats.get("manipulations_found", 0)
        dist_count = self.amd_stats.get("distributions_found", 0)
        dist_conf_count = self.amd_stats.get("distributions_with_bos_confluence", 0)
        bars_total = self.amd_stats.get("manip_to_dist_bars_total", 0)
        bars_count = self.amd_stats.get("manip_to_dist_count", 0)

        amd_conformity = {
            "consolidations_found": consol_count,
            "manipulations_found": manip_count,
            "distributions_found": dist_count,
            "distributions_with_bos_confluence": dist_conf_count,
            "consolidation_to_manipulation_pct": round((manip_count / consol_count) * 100, 2) if consol_count > 0 else 0.0,
            "manipulation_to_distribution_pct": round((dist_count / manip_count) * 100, 2) if manip_count > 0 else 0.0,
            "distribution_with_bos_confluence_pct": round((dist_conf_count / dist_count) * 100, 2) if dist_count > 0 else 0.0,
            "avg_bars_manipulation_to_distribution": round((bars_total / bars_count), 2) if bars_count > 0 else 0.0,
        }

        # Confidence tier breakdown
        from collections import defaultdict
        tier_stats = defaultdict(lambda: {"count": 0, "wins": 0, "pnl": 0.0})
        for trade in self.trades:
            tier = getattr(trade, "confidence_tier", "") or "base"
            tier_stats[tier]["count"] += 1
            if trade.r_multiple > 0:
                tier_stats[tier]["wins"] += 1
            tier_stats[tier]["pnl"] += trade.net_pnl
        # Compute win rates
        confidence_tier_stats = {}
        for tier_name, stats in tier_stats.items():
            wr = (stats["wins"] / stats["count"] * 100) if stats["count"] > 0 else 0
            confidence_tier_stats[tier_name] = {
                "count": stats["count"],
                "wins": stats["wins"],
                "win_rate": round(wr, 1),
                "pnl": round(stats["pnl"], 2),
            }

        return {
            "backtest_id": self.backtest_id,
            "initial_capital": self.initial_capital,
            "final_capital": self.current_balance,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate * 100, 2),
            "avg_r_multiple": round(avg_r, 3),
            "avg_win_r": round(avg_win_r, 3),
            "avg_loss_r": round(avg_loss_r, 3),
            "expectancy_r": round(expectancy, 3),
            # P&L breakdown
            "gross_pnl_usd": round(total_gross_pnl, 2),
            "net_pnl_usd": round(total_net_pnl, 2),
            "total_pnl_usd": round(total_net_pnl, 2),  # For compatibility
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 3),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            # Exit breakdown
            "sl_exits": sl_exits,
            "tp_exits": tp_exits,
            "rollover_exits": rollover_exits,
            "timeout_exits": timeout_exits,
            # Curves
            "equity_curve": self.equity_curve,
            "mtm_equity_curve": self.mtm_equity_curve,
            "trades": trades_df.to_dict("records"),
            # Validation
            "validation": {
                "meets_min_trades": total_trades >= VALIDATION["min_trades"],
                "meets_expectancy": expectancy >= VALIDATION["min_expectancy_r"],
                "meets_drawdown": max_drawdown <= VALIDATION["max_drawdown_pct"],
                # Objective profile for small-capital optimization branch
                "meets_trade_band_300_500": 300 <= total_trades <= 500,
                "meets_net_positive": total_net_pnl > 0,
                "objective_pass": (300 <= total_trades <= 500) and (total_net_pnl > 0),
            },
            # Stats
            "funnel_stats": self.rejection_stats,
            "confluence_stats": self.confluence_stats,
            "cost_stats": self.cost_stats,
            "amd_conformity": amd_conformity,
            "confidence_tier_stats": confidence_tier_stats,
        }

    def save_trades_to_db(self, db: Database):
        """Save all trades to database."""
        if not self.trades:
            return

        db_trades = []
        for t in self.trades:
            db_trade = Trade(
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                direction=t.direction,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                sl_price=t.sl_price,
                tp_price=t.tp_price,
                position_size=t.position_size,
                r_multiple=t.r_multiple,
                pnl_pips=t.pnl_pips,
                pnl_usd=t.net_pnl,  # Use net P&L
                consolidation_high=t.consolidation_high,
                consolidation_low=t.consolidation_low,
                manipulation_extreme=t.manipulation_extreme,
                manipulation_direction=t.manipulation_direction,
                backtest_id=t.backtest_id,
            )
            db_trades.append(db_trade)

        db.insert_trades(db_trades)
        logger.info(f"Saved {len(db_trades)} trades to database")
