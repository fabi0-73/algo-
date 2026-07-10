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

from config import STRATEGY, BACKTEST, VALIDATION, EXECUTION, TIME_CONFIG, SESSION_FILTER, PHANTOM_FILLS, MARKET_CHASE, RISK_MODEL, ADAPTIVE_EXITS, DRAWDOWN_CONTROLS, SIGNAL_CONFIDENCE, SWEEP_MODEL, NY_IB_MODEL, validate_config
from src.strategy.liquidity_levels import add_asian_range, find_swing_points, get_active_levels
from src.strategy.sweep_entry import detect_sweep_at_candle
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
    calculate_confluence_score,
    calculate_move_potential,
    calculate_signal_confidence,
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
from src.backtest.execution import ExecutionEngine, FillResult, ExitDecision, CostBreakdown

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
    swap_cost: float = 0.0
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
    worst_price_against: float = 0.0  # Worst low (LONG) or worst high (SHORT) since entry
    trailing_active: bool = False     # Whether trailing stop has activated
    original_tp: float = 0.0         # Original TP before trailing disables it

    # MFE/MAE (Maximum Favorable/Adverse Excursion) in R-multiples
    mfe_r: float = 0.0
    mae_r: float = 0.0

    # Move potential score (0-5) - predicts how far the setup may run
    move_potential: int = 0

    # Empirical signal confidence (0-4 score + LOW/MODERATE/GOOD/HIGH label)
    signal_confidence: int = 0
    confidence_label: str = ""

    # Entry model provenance + per-trade exit style
    entry_model: str = "AMD"       # "AMD" | "SWEEP"
    exit_style: str = ""           # "" (global behavior) | "FIXED_RR" | "HYBRID"
    sweep_level_kind: str = ""     # which liquidity level was swept (SWEEP trades)

    # Adaptive exit tier (when ADAPTIVE_EXITS enabled)
    exit_tier: str = ""                     # "runner", "standard", or "" (fallthrough)
    tier_tp_rr: float = 0.0
    tier_trailing_activation_r: float = 0.0
    tier_trailing_atr_mult: float = 0.0
    tier_be_trigger_r: float = 0.0
    tier_be_buffer_atr_mult: float = 0.0

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
    # active_patterns removed — was defined but never populated or cleaned

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
        enable_phantom_fills: bool = None,
        enable_market_chase: bool = None,
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
        # Phase 2 drawdown-control state
        self.equity_peak = self.initial_capital
        self.consecutive_losses = 0
        self.dd_halted = False
        self.dd_halt_start_idx = None

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

        # Phantom fills analysis
        self.phantom_fills_enabled = enable_phantom_fills if enable_phantom_fills is not None else PHANTOM_FILLS.get("enabled", False)
        self.phantom_trades: List[dict] = []

        # Market chase — enter at close for high-quality missed fills
        self.market_chase_enabled = enable_market_chase if enable_market_chase is not None else MARKET_CHASE.get("enabled", False)
        self.chase_stats = {
            "chase_attempts": 0,
            "chase_executed": 0,
            "chase_rejected_direction": 0,
            "chase_rejected_confluence": 0,
            "chase_rejected_slippage": 0,
            "chase_rejected_risk_invalid": 0,
        }

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
            "filtered_monthly_limit": 0,
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
            "total_swap_cost": 0.0,
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
        """Pre-compute midnight (00:00 EST) open prices keyed by date.

        Replaces the per-manipulation 300-bar backward scan with a single
        forward pass over all timestamps, yielding O(1) lookups later.
        Uses _MIDNIGHT_HOUR from manipulation module to handle non-UTC data.
        """
        from src.strategy.manipulation import _MIDNIGHT_HOUR
        result = {}
        ts_arr = self._timestamps
        opens_arr = self._opens
        for i in range(len(ts_arr)):
            ts = pd.Timestamp(ts_arr[i])
            if hasattr(ts, "hour") and ts.hour == _MIDNIGHT_HOUR and ts.minute == 0:
                result[ts.date()] = float(opens_arr[i])
        return result

    def run(self, df: pd.DataFrame, verbose: bool = False) -> Dict[str, Any]:
        """Run backtest on historical data."""
        # Validate configuration for conflicting settings
        config_warnings = validate_config()
        for w in config_warnings:
            logger.warning(f"Config warning: {w}")

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

        # Sweep model: Asian range levels + swing points for equal-level clustering
        self._sweep_mode = SWEEP_MODEL.get("strategy_mode", "AMD") if SWEEP_MODEL.get("enabled", False) else "AMD"
        if self._sweep_mode in ("SWEEP", "BOTH"):
            df = add_asian_range(df)

        # NY_IB model: per-day initial-balance state (see _scan_for_ny_ib)
        self._nyib_enabled = NY_IB_MODEL.get("enabled", False)
        self._nyib_only = self._nyib_enabled and NY_IB_MODEL.get("only", False)
        self._nyib_reset_day(None)

        # Pre-extract numpy arrays for the entire dataset (avoids millions of df[col].values calls)
        self._highs = df["high"].values
        self._lows = df["low"].values
        self._opens = df["open"].values
        self._closes = df["close"].values
        self._timestamps = df["timestamp"].values
        self._atrs = df["atr"].values
        self._tick_volumes = df["tick_volume"].values if "tick_volume" in df.columns else None

        if self._sweep_mode in ("SWEEP", "BOTH"):
            self._volumes = df["volume"].values if "volume" in df.columns else self._tick_volumes
            self._swing_high_idx, self._swing_low_idx = find_swing_points(
                self._highs, self._lows, int(SWEEP_MODEL.get("swing_strength", 3))
            )
            self._lv_cols = {
                k: (df[k].values if k in df.columns else None)
                for k in ("prev_day_high", "prev_day_low", "prev_week_high",
                          "prev_week_low", "asian_high", "asian_low")
            }
            self._seen_sweeps = set()

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
        # Phase 2 drawdown-control state
        self.equity_peak = self.initial_capital
        self.consecutive_losses = 0
        self.dd_halted = False
        self.dd_halt_start_idx = None

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
            "filtered_monthly_limit": 0,
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

        # Reset adaptive confidence recalibration state (E-adaptive)
        self._adaptive_gated = {}   # bucket label -> trade_n when gated
        self.adaptive_events = []

        # Reset cost stats
        self.cost_stats = {
            "total_spread_cost": 0.0,
            "total_slippage_cost": 0.0,
            "total_commission_cost": 0.0,
            "total_swap_cost": 0.0,
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
                if self._sweep_mode != "SWEEP" and not self._nyib_only:
                    self._scan_for_patterns(df, i, verbose)
                if not self.state.in_position and self._sweep_mode in ("SWEEP", "BOTH"):
                    self._scan_for_sweep(df, i, verbose)
                if not self.state.in_position and self._nyib_enabled:
                    self._scan_for_ny_ib(df, i, verbose)
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
            round(consol.range_high, 4),
            round(consol.range_low, 4),
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
        bypass_session_window: bool = False,
        bypass_htf_bias: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check all filters for entry permission.

        bypass_session_window: skip ONLY killzone/asian-session rejections
        (for models whose window is outside AMD killzones, e.g. NY_IB).
        Blackouts, loss limits, cooldown, rollover and news still apply.

        Returns:
            Tuple of (can_enter, rejection_reason)
        """
        ts = row.get("timestamp")
        atr = row.get("atr", 0.0)

        # 0. Drawdown circuit breaker (account-level; mirrors live kill switch).
        #    Blocks new entries while halted; resumes after equity recovers (hysteresis)
        #    or after halt_max_bars (deadlock fix).
        if self._drawdown_halted(current_idx):
            self.rejection_stats["filtered_drawdown_halt"] = self.rejection_stats.get("filtered_drawdown_halt", 0) + 1
            return False, "drawdown_circuit_breaker"

        # 1. Session/Time filter (kill zone, Asian session, daily/monthly loss limits).
        #    Basis = current equity (compounding-consistent) rather than fixed initial capital.
        can_enter, reason = self.time_filter.can_enter_trade(ts, self.current_balance)
        if not can_enter and bypass_session_window and ("killzone" in reason or "asian" in reason):
            can_enter = True  # window gate waived for this model; all else still applies
        if not can_enter:
            if reason == "blackout_weekday":
                self.rejection_stats["filtered_blackout_weekday"] += 1
            elif "blackout" in reason:
                self.rejection_stats["filtered_blackout_hour"] += 1
            elif "killzone" in reason or "asian" in reason:
                self.rejection_stats["filtered_session"] += 1
            elif "monthly_loss_limit" in reason:
                self.rejection_stats["filtered_monthly_limit"] += 1
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
        if not bypass_htf_bias:
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

    def _current_drawdown(self) -> float:
        """Realized-equity drawdown from the running peak (0.0 .. 1.0)."""
        if self.equity_peak > 0:
            return max(0.0, (self.equity_peak - self.current_balance) / self.equity_peak)
        return 0.0

    def _drawdown_halted(self, current_idx: int = None) -> bool:
        """Account-level circuit breaker with hysteresis (mirrors live kill switch).

        Halts NEW entries once drawdown from the equity peak reaches
        max_account_dd_pct; resumes when DD recovers to resume_dd_pct OR after
        halt_max_bars (deadlock fix: with entries blocked and no position open,
        equity is frozen and could otherwise never recover — the risk-scaling
        floor keeps the resumed size halved while DD remains deep).
        """
        dc = DRAWDOWN_CONTROLS
        if not (dc.get("enabled", False) and dc.get("circuit_breaker_enabled", False)):
            return False
        dd = self._current_drawdown()
        if not self.dd_halted and dd >= dc.get("max_account_dd_pct", 1.0):
            self.dd_halted = True
            self.dd_halt_start_idx = current_idx
        elif self.dd_halted:
            timed_out = (
                current_idx is not None
                and self.dd_halt_start_idx is not None
                and current_idx - self.dd_halt_start_idx >= dc.get("halt_max_bars", 10**9)
            )
            if dd <= dc.get("resume_dd_pct", 0.0) or timed_out:
                self.dd_halted = False
                self.dd_halt_start_idx = None
        return self.dd_halted

    def _risk_scale_factor(self) -> float:
        """Multiplier (0,1] applied to risk_pct: de-risk in drawdown / after loss streaks."""
        dc = DRAWDOWN_CONTROLS
        if not dc.get("enabled", False):
            return 1.0
        factor = 1.0
        # Equity-based scaling: linear from 1.0 at start_dd down to min_factor at full_dd
        if dc.get("risk_scaling_enabled", False):
            dd = self._current_drawdown()
            start = dc.get("risk_scale_start_dd", 0.06)
            full = dc.get("risk_scale_full_dd", 0.15)
            floor = dc.get("risk_scale_min_factor", 0.5)
            if dd > start and full > start:
                frac = min(1.0, (dd - start) / (full - start))
                factor = 1.0 - frac * (1.0 - floor)
        # Consecutive-loss brake
        if dc.get("consec_loss_enabled", False) and self.consecutive_losses >= dc.get("max_consecutive_losses", 10**9):
            factor = min(factor, dc.get("consec_loss_risk_factor", 0.5))
        return factor

    def _get_confidence_tier(self, confluence_score: int, entry_hour: int) -> tuple:
        """Return (tier_name, risk_pct, lot_bonus) based on confluence score and entry hour."""
        from config import CONFIDENCE_SIZING, RISK_MODEL

        if not CONFIDENCE_SIZING.get("enabled", False):
            return ("base", RISK_MODEL["risk_pct_per_trade_default"], 0.0)

        prime_start = CONFIDENCE_SIZING.get("prime_hours_start", 13)
        prime_end = CONFIDENCE_SIZING.get("prime_hours_end", 17)
        is_prime = prime_start <= entry_hour <= prime_end

        for tier in CONFIDENCE_SIZING.get("tiers", []):
            if confluence_score >= tier["min_confluence_score"]:
                if tier.get("prime_hours_only", False) and not is_prime:
                    continue
                return (tier["name"], tier["risk_pct"], tier.get("lot_bonus", 0.0))

        return ("base", CONFIDENCE_SIZING.get("base_risk_pct", 0.003), 0.0)

    def _resolve_exit_tier(self, move_potential: int) -> dict:
        """Return the matching adaptive exit tier dict, or None for fallthrough."""
        for tier in ADAPTIVE_EXITS.get("tiers", []):
            if move_potential >= tier.get("min_move_potential", 999):
                return tier
        return None

    def _get_tier_config(self, tier_name: str, key: str, default=None):
        """Look up a config value from the named adaptive exit tier."""
        for tier in ADAPTIVE_EXITS.get("tiers", []):
            if tier.get("name") == tier_name:
                return tier.get(key, default)
        return default

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
        for consol_end_offset in range(min_offset, max_offset,
                                       STRATEGY.get("consolidation_scan_step", 2)):
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
            if not validate_distribution_strength(df, dist, min_follow_through_candles=follow_through_candles,
                                                  current_idx=current_idx):
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
                    current_idx=current_idx,
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
            # Use ATR-based tolerance to match how equal levels are detected
            sweep_tolerance = atr * 0.05
            equal_level_swept = False
            if manip.direction == "UP":
                equal_level_swept = bool(
                    consol.has_equal_highs
                    and consol.equal_high_level > 0
                    and manip.extreme_price >= consol.equal_high_level - sweep_tolerance
                )
            elif manip.direction == "DOWN":
                equal_level_swept = bool(
                    consol.has_equal_lows
                    and consol.equal_low_level > 0
                    and manip.extreme_price <= consol.equal_low_level + sweep_tolerance
                )

            volume_confirmed = getattr(manip, "volume_confirmed", False)
            confluence_score = calculate_confluence_score(
                bos_confirmed=(bos is not None and bos.valid),
                fvg_at_level=bool(fvg_at_level),
                ob_at_level=bool(ob_at_level),
                equal_level_swept=equal_level_swept,
                volume_confirmed=volume_confirmed,
            )

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
            confidence_tier, tier_risk_pct, lot_bonus = self._get_confidence_tier(confluence_score, entry_hour)

            # Phase 2: de-risk in drawdown / after loss streaks (scales the tier's risk_pct).
            risk_scale = self._risk_scale_factor()
            if risk_scale < 1.0:
                tier_risk_pct *= risk_scale
                if DRAWDOWN_CONTROLS.get("suppress_lot_bonus_when_scaling", True):
                    lot_bonus = 0.0

            # Compute move potential score for adaptive exit management
            move_potential = calculate_move_potential(
                velocity_score=getattr(manip, "velocity_score", 0.0),
                session_hour=entry_hour,
                body_expansion=getattr(dist, "body_expansion", 0.0),
                consolidation_quality=score_consolidation_quality(consol),
                equal_level_swept=equal_level_swept,
            )
            self._pending_move_potential = move_potential

            # Empirical signal confidence (0-4 + label) — display on every trade;
            # optionally up-size the HIGH bucket (validated: GOOD/HIGH hold ~1.1-1.2R OOS).
            signal_conf, conf_label = calculate_signal_confidence(
                confluence_score, move_potential, entry_hour
            )
            self._pending_signal_confidence = signal_conf
            self._pending_confidence_label = conf_label

            # Confidence gate: skip entries below the minimum empirical confidence
            # score (0 = disabled). LOW bucket wins ~37% vs HIGH ~54% — culling the
            # bottom raises WR and per-trade expectancy at the cost of trade count.
            min_conf = SIGNAL_CONFIDENCE.get("min_confidence_to_trade", 0)
            if min_conf > 0 and signal_conf < min_conf:
                self.rejection_stats.setdefault("low_signal_confidence_gate", 0)
                self.rejection_stats["low_signal_confidence_gate"] += 1
                if verbose:
                    logger.debug(
                        f"Confidence gate: score {signal_conf} < {min_conf} ({conf_label})"
                    )
                continue

            # E-adaptive: bucket currently gated by rolling recalibration
            if conf_label in self._adaptive_gated:
                self.rejection_stats.setdefault("adaptive_confidence_gate", 0)
                self.rejection_stats["adaptive_confidence_gate"] += 1
                continue

            if (SIGNAL_CONFIDENCE.get("enabled", False)
                    and SIGNAL_CONFIDENCE.get("size_by_confidence", False)):
                # Respect drawdown de-risking: no up-sizing while risk-scaled down.
                suppressed = risk_scale < 1.0 and DRAWDOWN_CONTROLS.get(
                    "suppress_lot_bonus_when_scaling", True)
                if not suppressed:
                    if conf_label == "HIGH":
                        lot_bonus += SIGNAL_CONFIDENCE.get("high_extra_lots", 0.0)
                    elif conf_label == "GOOD":
                        lot_bonus += SIGNAL_CONFIDENCE.get("good_extra_lots", 0.0)

            # Calculate risk with confidence-based sizing
            risk = calculate_risk(entry, self.current_balance, self.min_rr, self.max_risk_pct,
                                  risk_pct_override=tier_risk_pct)
            if not risk.valid:
                self.rejection_stats["risk_invalid"] += 1
                if verbose and risk.rejection_reason:
                    logger.debug(f"Risk rejected: {risk.rejection_reason}")
                continue

            # Apply lot bonus for higher-tier trades (bypasses lot rounding issue)
            if lot_bonus > 0:
                from config import RISK_MODEL
                new_lots = risk.position_size + lot_bonus
                new_lots = min(new_lots, RISK_MODEL["max_lot"])
                risk.position_size = new_lots
                risk.risk_amount_usd = round(risk.risk_per_lot_usd * new_lots, 2)
                risk.notional_usd = round(entry.entry_price * RISK_MODEL["contract_size"] * new_lots, 2)

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

                # Attempt market chase for high-quality missed fills
                if self.market_chase_enabled:
                    chased = self._attempt_market_chase(
                        entry, risk, row, current_idx, df, atr,
                        confidence_tier, tier_risk_pct, pattern_key, verbose,
                    )
                    if chased:
                        return  # Real trade created — exit scan loop

                if self.phantom_fills_enabled:
                    self._capture_phantom_trade(
                        entry, risk, row, current_idx, df, atr,
                        confidence_tier, tier_risk_pct, verbose,
                    )
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
            self._pending_entry_model = "AMD"
            self._pending_exit_style = ""
            self._pending_sweep_kind = ""
            self._execute_entry(entry, risk, fill_result, row, current_idx, df, verbose)

            # Record trade for session tracking
            ts = row.get("timestamp")
            self.time_filter.record_trade(ts)

            return  # Only one entry per bar

    def _scan_for_sweep(self, df: pd.DataFrame, current_idx: int, verbose: bool):
        """Sweep-reversal entry path (SWEEP_MODEL) — runs beside the AMD scanner.

        Fades a liquidity sweep of an OHLCV-derived stop-cluster level. Shares
        _check_filters (session/news/HTF/drawdown breaker) and the execution/exit
        plumbing; trades are tagged entry_model="SWEEP" with a per-trade exit style.
        """
        i = current_idx
        lookback_needed = int(SWEEP_MODEL.get("level_lookback", 200))
        if i < lookback_needed:
            return
        atr = self._atrs[i]
        if not atr or np.isnan(atr) or atr <= 0:
            return

        row_levels = {k: (arr[i] if arr is not None else None)
                      for k, arr in self._lv_cols.items()}
        levels = get_active_levels(
            i, float(self._closes[i]), float(atr), row_levels,
            swing_high_idx=self._swing_high_idx, swing_low_idx=self._swing_low_idx,
            highs=self._highs, lows=self._lows, cfg=SWEEP_MODEL,
        )
        if not levels:
            return

        signals = detect_sweep_at_candle(
            i, self._highs, self._lows, self._opens, self._closes,
            self._volumes, float(atr), levels, cfg=SWEEP_MODEL,
        )
        if not signals:
            return

        for sig in signals:
            key = (sig.direction, round(sig.level_price, 1), sig.poke_bar_idx)
            if key in self._seen_sweeps:
                continue
            self._seen_sweeps.add(key)
            self.rejection_stats["sweep_signals"] = self.rejection_stats.get("sweep_signals", 0) + 1

            if sig.direction == "SHORT" and (
                SWEEP_MODEL.get("long_only", False)
                or not STRATEGY.get("allow_short_trades", True)
            ):
                continue

            row = df.iloc[i]
            close_i = float(self._closes[i])
            entry = EntrySignal(
                valid=True,
                direction=sig.direction,
                entry_price=close_i,
                entry_candle_idx=i,
                entry_timestamp=row.get("timestamp"),
                rejection_confirmed=sig.rejection_confirmed,
                retest_level=sig.level_price,
                manipulation_extreme=sig.sweep_extreme,
                manipulation_direction="UP" if sig.direction == "SHORT" else "DOWN",
                entry_mode="SWEEP",
            )
            entry.desired_entry_price = close_i  # enter at signal close, not at the level

            can_enter, filter_reason = self._check_filters(df, i, entry, row)
            if not can_enter:
                self.rejection_stats["sweep_filtered"] = self.rejection_stats.get("sweep_filtered", 0) + 1
                if verbose:
                    logger.debug(f"Sweep entry filtered: {filter_reason}")
                continue

            # Base-tier risk (no AMD confluence for sweeps), drawdown-scaled like AMD
            entry_hour = row.get("timestamp").hour if hasattr(row.get("timestamp"), "hour") else 0
            confidence_tier, tier_risk_pct, _ = self._get_confidence_tier(0, entry_hour)
            risk_scale = self._risk_scale_factor()
            if risk_scale < 1.0:
                tier_risk_pct *= risk_scale

            tp_rr = float(SWEEP_MODEL.get("tp_rr", 2.5))
            risk = calculate_risk(entry, self.current_balance, tp_rr, self.max_risk_pct,
                                  atr=float(atr), risk_pct_override=tier_risk_pct)
            if not risk.valid:
                self.rejection_stats["sweep_risk_invalid"] = self.rejection_stats.get("sweep_risk_invalid", 0) + 1
                if verbose and risk.rejection_reason:
                    logger.debug(f"Sweep risk rejected: {risk.rejection_reason}")
                continue

            fill_result = self.execution.simulate_entry_fill(
                entry=entry, candle=row, atr=atr, position_size=risk.position_size,
            )
            if not fill_result.filled:
                self.rejection_stats["fill_not_triggered"] += 1
                continue

            self.rejection_stats["entries_executed"] += 1
            self.rejection_stats["sweep_entries"] = self.rejection_stats.get("sweep_entries", 0) + 1
            self._pending_confidence_tier = confidence_tier
            self._pending_risk_pct = tier_risk_pct
            self._pending_move_potential = 0
            self._pending_signal_confidence = 0
            self._pending_confidence_label = ""     # AMD-calibrated ladder doesn't apply
            self._pending_entry_model = "SWEEP"
            self._pending_exit_style = SWEEP_MODEL.get("exit_style", "FIXED_RR")
            self._pending_sweep_kind = sig.level_kind
            self._execute_entry(entry, risk, fill_result, row, i, df, verbose)

            self.time_filter.record_trade(row.get("timestamp"))
            if verbose:
                logger.info(
                    f"SWEEP ENTRY: {sig.direction} @ {fill_result.fill_price:.2f} | "
                    f"level {sig.level_kind} {sig.level_price:.2f} | poke {sig.poke_atr_mult:.2f} ATR | "
                    f"exit {self._pending_exit_style}"
                )
            return  # one entry per bar

    def _nyib_reset_day(self, day):
        """Reset NY_IB per-day state; parse config windows once (day=None at init)."""
        if day is None:
            def _mins(s):
                hh, mm = (int(x) for x in str(s).split(":"))
                return hh * 60 + mm
            self._nyib_mins = (
                _mins(NY_IB_MODEL.get("ib_start", "16:30")),
                _mins(NY_IB_MODEL.get("ib_end", "17:30")),
                _mins(NY_IB_MODEL.get("scan_end", "22:00")),
                _mins(NY_IB_MODEL.get("eod_flat", "23:00")),
            )
        self._nyib_day = day
        self._nyib_final = False      # IB build attempted for this day
        self._nyib_hi = 0.0
        self._nyib_lo = 0.0
        self._nyib_valid = False
        self._nyib_attempted = False  # breakout attempt consumed
        self._nyib_order = None       # pending limit dict
        self._nyib_done = False       # trade taken today

    def _scan_for_ny_ib(self, df: pd.DataFrame, current_idx: int, verbose: bool):
        """NY Initial Balance pullback (NY_IB_MODEL) — runs beside the AMD scanner.

        Builds the first-NY-hour range from completed bars, then after a close
        beyond it (first per day) rests a LIMIT back inside the range with a
        pure BRACKET exit (small TP past the edge, SL across the range) and a
        23:00 force-flat. Lab-validated in src/research/strategies/ny_ib.py
        (runs 65fecd3b/6ef3f2fb); this branch re-validates under engine fills,
        news filter, and drawdown controls.

        Divergence from the lab (documented): the engine scans only while
        flat, so a breakout close that happens during an open AMD position is
        unseen and a LATER confirming close may arm the order instead of the
        literal first one.
        """
        i = current_idx
        ts = pd.Timestamp(self._timestamps[i])
        mins = ts.hour * 60 + ts.minute
        day = ts.normalize()
        if day != self._nyib_day:
            self._nyib_reset_day(day)

        ib_start, ib_end, scan_end, eod = self._nyib_mins
        if mins < ib_end:
            return

        # Expire the pending order at force-flat time
        if self._nyib_order is not None and mins >= eod:
            self._nyib_order = None
            return

        # Finalize today's IB once, walking back over completed bars
        if not self._nyib_final:
            self._nyib_final = True
            hi, lo, nb = -np.inf, np.inf, 0
            ref_close = None
            j = i - 1
            while j >= 0:
                tj = pd.Timestamp(self._timestamps[j])
                if tj.normalize() != day:
                    break
                mj = tj.hour * 60 + tj.minute
                if mj < ib_start:
                    break
                if ib_start <= mj < ib_end:
                    hi = max(hi, float(self._highs[j]))
                    lo = min(lo, float(self._lows[j]))
                    nb += 1
                    if ref_close is None:
                        ref_close = float(self._closes[j])  # last IB bar close
                j -= 1
            if nb < int(NY_IB_MODEL.get("min_ib_bars", 10)):
                self.rejection_stats["nyib_too_few_bars"] = self.rejection_stats.get("nyib_too_few_bars", 0) + 1
                return
            size = hi - lo
            if not (NY_IB_MODEL.get("ib_min_pct", 0.004) * ref_close
                    <= size <= NY_IB_MODEL.get("ib_max_pct", 0.02) * ref_close):
                self.rejection_stats["nyib_ib_size"] = self.rejection_stats.get("nyib_ib_size", 0) + 1
                return
            self._nyib_hi, self._nyib_lo, self._nyib_valid = hi, lo, True

        if not self._nyib_valid or self._nyib_done:
            return

        row = None

        # 1) Pending order: honest touch check, fill at the limit price
        if self._nyib_order is not None:
            o = self._nyib_order
            touched = (self._lows[i] <= o["limit"] if o["direction"] == "LONG"
                       else self._highs[i] >= o["limit"])
            if not touched:
                return
            atr = self._atrs[i]
            if not atr or np.isnan(atr) or atr <= 0:
                return
            row = df.iloc[i]
            entry = EntrySignal(
                valid=True,
                direction=o["direction"],
                entry_price=o["limit"],
                entry_candle_idx=i,
                entry_timestamp=row.get("timestamp"),
                retest_level=o["limit"],
                manipulation_extreme=o["sl"],
                manipulation_direction="DOWN" if o["direction"] == "LONG" else "UP",
                entry_mode="NY_IB",
                consolidation_high=self._nyib_hi,
                consolidation_low=self._nyib_lo,
            )
            entry.desired_entry_price = o["limit"]

            # Flat base-tier sizing (like SWEEP), drawdown-scaled; bypass
            # calculate_risk — its min_rr gate rejects RR ~0.2 by design.
            tier, tier_risk_pct, _ = self._get_confidence_tier(0, ts.hour)
            scale = self._risk_scale_factor()
            if scale < 1.0:
                tier_risk_pct *= scale
            stop_dist = abs(o["limit"] - o["sl"])
            contract = RISK_MODEL.get("contract_size", 100)
            step = RISK_MODEL.get("lot_step", 0.01)
            lots = (self.current_balance * tier_risk_pct) / (stop_dist * contract)
            lots = round(lots / step) * step
            lots = min(max(lots, RISK_MODEL.get("min_lot", 0.01)),
                       RISK_MODEL.get("max_lot", 1.0))
            reward = abs(o["tp"] - o["limit"])
            risk = RiskParams(
                valid=True, stop_loss=o["sl"], take_profit=o["tp"],
                stop_distance=stop_dist, reward_distance=reward,
                risk_reward_ratio=reward / stop_dist if stop_dist > 0 else 0.0,
                position_size=lots,
                risk_amount_usd=stop_dist * contract * lots,
                risk_per_lot_usd=stop_dist * contract,
            )
            fill_result = self.execution.simulate_entry_fill(
                entry=entry, candle=row, atr=atr, position_size=lots,
            )
            if not fill_result.filled:
                self.rejection_stats["fill_not_triggered"] += 1
                return
            self.rejection_stats["entries_executed"] += 1
            self.rejection_stats["nyib_entries"] = self.rejection_stats.get("nyib_entries", 0) + 1
            self._pending_confidence_tier = tier
            self._pending_risk_pct = tier_risk_pct
            self._pending_move_potential = 0
            self._pending_signal_confidence = 0
            self._pending_confidence_label = ""
            self._pending_entry_model = "NY_IB"
            self._pending_exit_style = NY_IB_MODEL.get("exit_style", "BRACKET")
            self._pending_sweep_kind = ""
            self._execute_entry(entry, risk, fill_result, row, i, df, verbose)
            self.time_filter.record_trade(row.get("timestamp"))
            self._nyib_done = True
            self._nyib_order = None
            if verbose:
                logger.info(
                    f"NY_IB ENTRY: {o['direction']} @ {fill_result.fill_price:.2f} | "
                    f"IB {self._nyib_lo:.2f}-{self._nyib_hi:.2f} | "
                    f"SL {o['sl']:.2f} TP {o['tp']:.2f}"
                )
            return

        # 2) Breakout detection (first confirming close consumes the attempt)
        if self._nyib_attempted or not (ib_end <= mins < scan_end):
            return
        close_i = float(self._closes[i])
        if close_i > self._nyib_hi:
            direction = "LONG"
        elif close_i < self._nyib_lo and not NY_IB_MODEL.get("long_only", False) \
                and STRATEGY.get("allow_short_trades", True):
            direction = "SHORT"
        else:
            return
        self._nyib_attempted = True
        self.rejection_stats["nyib_breakouts"] = self.rejection_stats.get("nyib_breakouts", 0) + 1

        size = self._nyib_hi - self._nyib_lo
        rf = NY_IB_MODEL.get("retrace_frac", 0.10)
        slm = NY_IB_MODEL.get("sl_range_mult", 1.0)
        tpm = NY_IB_MODEL.get("tp_range_mult", 0.20)
        if direction == "LONG":
            limit = self._nyib_hi - rf * size
            sl = limit - slm * size
            tp = self._nyib_hi + tpm * size
        else:
            limit = self._nyib_lo + rf * size
            sl = limit + slm * size
            tp = self._nyib_lo - tpm * size

        row = df.iloc[i]
        probe = EntrySignal(
            valid=True, direction=direction, entry_price=limit,
            entry_candle_idx=i, entry_timestamp=row.get("timestamp"),
            retest_level=limit, manipulation_extreme=sl,
            manipulation_direction="DOWN" if direction == "LONG" else "UP",
            entry_mode="NY_IB",
            consolidation_high=self._nyib_hi, consolidation_low=self._nyib_lo,
        )
        probe.desired_entry_price = limit
        can_enter, filter_reason = self._check_filters(
            df, i, probe, row,
            bypass_session_window=NY_IB_MODEL.get("bypass_session_window", True),
            bypass_htf_bias=NY_IB_MODEL.get("bypass_htf_bias", True),
        )
        if not can_enter:
            self.rejection_stats["nyib_filtered"] = self.rejection_stats.get("nyib_filtered", 0) + 1
            if verbose:
                logger.debug(f"NY_IB breakout filtered: {filter_reason}")
            return
        self._nyib_order = {"direction": direction, "limit": limit, "sl": sl, "tp": tp}
        self.rejection_stats["nyib_orders_placed"] = self.rejection_stats.get("nyib_orders_placed", 0) + 1
        if verbose:
            logger.info(
                f"NY_IB ORDER: {direction} limit {limit:.2f} (IB {self._nyib_lo:.2f}-"
                f"{self._nyib_hi:.2f}, SL {sl:.2f}, TP {tp:.2f})"
            )

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
        worst_price = fill.fill_price  # Initialize for MAE tracking
        trade = TradeRecord(
            entry_time=current_candle.get("timestamp", datetime.now()),
            direction=entry.direction,
            entry_price=fill.fill_price,  # Use actual fill price
            sl_price=risk.stop_loss,
            tp_price=risk.take_profit,
            original_sl=risk.stop_loss,
            sl_moved_to_be=False,
            best_price_in_favor=best_price,
            worst_price_against=worst_price,
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
            move_potential=getattr(self, '_pending_move_potential', 0),
            signal_confidence=getattr(self, '_pending_signal_confidence', 0),
            confidence_label=getattr(self, '_pending_confidence_label', ''),
            entry_model=getattr(self, '_pending_entry_model', 'AMD'),
            exit_style=getattr(self, '_pending_exit_style', ''),
            sweep_level_kind=getattr(self, '_pending_sweep_kind', ''),
        )

        # Apply adaptive exit tier if enabled
        if ADAPTIVE_EXITS.get("enabled", False):
            tier = self._resolve_exit_tier(trade.move_potential)
            if tier:
                trade.exit_tier = tier["name"]
                trade.tier_tp_rr = tier.get("tp_rr", 0)
                trade.tier_trailing_activation_r = tier.get("trailing_activation_r", 0)
                trade.tier_trailing_atr_mult = tier.get("trailing_atr_mult", 0)
                trade.tier_be_trigger_r = tier.get("be_trigger_r", 0)
                trade.tier_be_buffer_atr_mult = tier.get("be_buffer_atr_mult", 0)
                # Adjust TP based on tier
                stop_dist = abs(trade.entry_price - trade.sl_price)
                if tier.get("tp_rr", 0) > 0 and stop_dist > 0:
                    if trade.direction == "LONG":
                        trade.tp_price = trade.entry_price + stop_dist * tier["tp_rr"]
                    else:
                        trade.tp_price = trade.entry_price - stop_dist * tier["tp_rr"]
                elif tier.get("tp_rr", 0) == 0:
                    # No static TP for runners — trail will manage exit
                    trade.original_tp = trade.tp_price
                    if trade.direction == "LONG":
                        trade.tp_price = float('inf')
                    else:
                        trade.tp_price = 0.0

        self.state.in_position = True
        self.state.current_trade = trade
        self.state.position_entry_idx = current_idx

        if verbose:
            tier_info = f" | ExitTier: {trade.exit_tier}" if trade.exit_tier else ""
            logger.info(
                f"ENTRY: {entry.direction} @ {fill.fill_price:.2f} | "
                f"SL: {risk.stop_loss:.2f} | TP: {risk.take_profit:.2f} | "
                f"Size: {risk.position_size:.2f} | "
                f"Costs: ${fill.costs.total_cost:.2f} | "
                f"Tier: {trade.confidence_tier} ({trade.risk_pct_used*100:.1f}%)"
                f"{tier_info}"
            )

    def _check_exit(self, df: pd.DataFrame, current_idx: int, candle: pd.Series, verbose: bool):
        """Check if current position should be exited (supports partial TP)."""
        from config import RISK_MODEL

        trade = self.state.current_trade
        ts = candle.get("timestamp")
        atr = candle.get("atr", 0.0)

        # Check timeout (extended for runner tier)
        bars_in_trade = current_idx - self.state.position_entry_idx
        max_duration = self.max_trade_duration
        if (ADAPTIVE_EXITS.get("enabled", False)
                and ADAPTIVE_EXITS.get("extend_duration_for_runners", False)
                and trade.exit_tier == "runner"):
            max_duration = ADAPTIVE_EXITS.get("runner_max_duration", 360)
        # E4 overlay: LONG runners with an active trailing stop may ride longer to
        # harvest the overnight drift (gold overnight returns positive, J Econ
        # Finance 2017). Risk is already trailing-protected; swap modeled honestly.
        long_runner_max = STRATEGY.get("long_runner_max_duration", 0)
        if (long_runner_max > 0 and trade.direction == "LONG"
                and getattr(trade, "trailing_active", False)):
            max_duration = max(max_duration, long_runner_max)
        if bars_in_trade >= max_duration:
            self._exit_position(candle["close"], "TIMEOUT", candle, verbose)
            return

        # Check rollover exit. E4 overlay: LONG runners may hold through rollover —
        # peer-reviewed gold overnight bias is positive for longs (J Econ Finance 2017);
        # swap cost is modeled honestly, so the backtest decides if the edge survives it.
        if self.time_filter.should_close_for_rollover(ts):
            hold_long = SESSION_FILTER.get("allow_long_overnight_hold", False)
            if not (hold_long and trade.direction == "LONG"):
                self._exit_position(candle["close"], "ROLLOVER", candle, verbose)
                return

        # NY_IB per-model force-flat (23:00 broker): never hold past the session
        if getattr(trade, "entry_model", "") == "NY_IB" and ts is not None:
            eod_min = self._nyib_mins[3] if hasattr(self, "_nyib_mins") else 23 * 60
            if ts.hour * 60 + ts.minute >= eod_min:
                self._exit_position(candle["close"], "EOD_FLAT", candle, verbose)
                return

        # --- Exit check FIRST, against stops as they stood at bar start ---
        # Causality: BE/trail moves computed from THIS bar's extremes take
        # effect on the NEXT bar (lab convention, live knowability) — a stop
        # raised by this bar's high must not stop out on this bar's low.
        use_adaptive = ADAPTIVE_EXITS.get("enabled", False) and trade.exit_tier
        be_buffer_mult = trade.tier_be_buffer_atr_mult if use_adaptive \
            else STRATEGY.get("be_buffer_atr_mult", 0.1)

        # Check if partial TP is enabled (via RISK_MODEL, adaptive tier, or per-trade HYBRID style)
        tier_partial = (use_adaptive and self._get_tier_config(trade.exit_tier, "partial_tp_enabled", False))
        hybrid_style = getattr(trade, "exit_style", "") == "HYBRID"
        bracket_style = getattr(trade, "exit_style", "") == "BRACKET"
        if (RISK_MODEL.get("partial_tp_enabled", False) or tier_partial or hybrid_style) and not bracket_style:
            if hybrid_style:
                partial_at_r = SWEEP_MODEL.get("hybrid_partial_at_r", 1.0)
                partial_pct = SWEEP_MODEL.get("hybrid_partial_pct", 0.5)
            else:
                partial_at_r = self._get_tier_config(trade.exit_tier, "partial_tp_at_r") if tier_partial else None
                partial_pct = self._get_tier_config(trade.exit_tier, "partial_close_pct") if tier_partial else None
            exit_decision = self.execution.check_exit_with_partial(
                trade=trade,
                candle=candle,
                atr=atr,
                partial_at_r=partial_at_r,
                partial_pct=partial_pct,
                be_buffer_atr=be_buffer_mult if (tier_partial or hybrid_style) else None,
            )
        else:
            exit_decision = self.execution.check_exit(
                trade=trade,
                candle=candle,
                atr=atr,
            )

        # Update best/worst price tracking (for trailing stop and MFE/MAE).
        # Runs after the exit decision (stops must not see this bar's extremes)
        # but before recording it, so excursions include the exit bar.
        if trade.direction == "LONG":
            trade.best_price_in_favor = max(
                getattr(trade, "best_price_in_favor", trade.entry_price),
                candle["high"],
            )
            trade.worst_price_against = min(
                trade.worst_price_against if trade.worst_price_against > 0 else trade.entry_price,
                candle["low"],
            )
        else:
            trade.best_price_in_favor = min(
                getattr(trade, "best_price_in_favor", trade.entry_price),
                candle["low"],
            )
            trade.worst_price_against = max(
                trade.worst_price_against if trade.worst_price_against > 0 else trade.entry_price,
                candle["high"],
            )

        # Handle partial close
        if exit_decision.is_partial:
            self._handle_partial_close(exit_decision, candle, verbose)
            return

        if exit_decision.should_exit:
            self._exit_position(exit_decision.exit_price, exit_decision.exit_reason, candle, verbose)
            return

        # --- No exit this bar: stage BE/trail/TP-disable for the NEXT bars ---
        # Move SL to breakeven at configured R level (before trailing stop)
        # Use per-trade tier params when adaptive exits are active
        be_trigger_r = trade.tier_be_trigger_r if use_adaptive else STRATEGY.get("move_sl_to_be_at_r", 0)
        # BRACKET style = pure fixed bracket: no BE move, no trailing, no partial
        if getattr(trade, "exit_style", "") == "BRACKET":
            be_trigger_r = 0
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
                    be_buffer = atr * be_buffer_mult
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
        # Use per-trade tier trailing params when adaptive exits are active
        trailing_enabled = STRATEGY.get("trailing_stop_enabled", False) or (use_adaptive and trade.tier_trailing_activation_r > 0)
        # Per-trade exit style: FIXED_RR/BRACKET trades ride to their fixed TP — never trail
        if getattr(trade, "exit_style", "") in ("FIXED_RR", "BRACKET"):
            trailing_enabled = False
        if trailing_enabled and atr and atr > 0:
            # Use original SL for R calculation so trailing works after BE move
            ref_sl = trade.original_sl if trade.original_sl > 0 else trade.sl_price
            stop_distance = (
                trade.entry_price - ref_sl
                if trade.direction == "LONG"
                else ref_sl - trade.entry_price
            )
            if stop_distance > 0:
                # Override trailing params from tier if adaptive exits active
                if use_adaptive and trade.tier_trailing_activation_r > 0:
                    trail_activation_r = trade.tier_trailing_activation_r
                    trail_atr_mult = trade.tier_trailing_atr_mult
                else:
                    trail_activation_r = STRATEGY.get("trailing_stop_activation_r", 1.5)
                    trail_atr_mult = STRATEGY.get("trailing_stop_atr_mult", 2.0)

                effective_sl, trailing_active = get_effective_stop_with_trailing(
                    entry_price=trade.entry_price,
                    current_sl=trade.sl_price,
                    current_extreme_price=trade.best_price_in_favor,
                    stop_distance=stop_distance,
                    atr=atr,
                    direction=trade.direction,
                    activation_r=trail_activation_r,
                    trail_atr_mult=trail_atr_mult,
                )
                if trailing_active and effective_sl != trade.sl_price:
                    trade.sl_price = effective_sl
                if trailing_active:
                    trade.trailing_active = True

        # Disable TP when trailing is active — let the trail manage the exit
        # For adaptive exit tiers, TP is already set at entry time
        if not use_adaptive and STRATEGY.get("disable_tp_when_trailing", False) and trade.trailing_active:
            if trade.original_tp == 0.0:
                trade.original_tp = trade.tp_price  # Save before overwriting
            if trade.direction == "LONG":
                trade.tp_price = float('inf')
            else:
                trade.tp_price = 0.0

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

        # Charge proportional costs on the partial close
        partial_pct = exit_decision.partial_close_pct
        partial_commission = trade.commission_cost * partial_pct
        partial_spread = trade.spread_cost * partial_pct
        partial_slippage = trade.slippage_cost * partial_pct
        partial_costs = partial_commission + partial_spread + partial_slippage
        net_partial_pnl = partial_pnl - partial_costs

        # Update trade record
        trade.partial_tp_taken = True
        trade.partial_tp_price = exit_decision.exit_price
        trade.partial_close_pct = partial_pct
        trade.partial_pnl = net_partial_pnl
        trade.remaining_size = trade.position_size * (1 - partial_pct)

        # Move SL to breakeven if configured
        if exit_decision.new_sl_price > 0:
            trade.sl_price = exit_decision.new_sl_price
            trade.sl_moved_to_be = True

        # Credit net partial PnL to balance
        self.current_balance += net_partial_pnl

        if verbose:
            logger.info(
                f"PARTIAL TP: {trade.direction} closed {exit_decision.partial_close_pct*100:.0f}% "
                f"@ {exit_decision.exit_price:.2f} | PnL: ${partial_pnl:.2f} | "
                f"New SL: {trade.sl_price:.2f} (BE)"
            )

    def _calculate_swap_cost(self, entry_time, exit_time, position_size: float) -> float:
        """Overnight swap cost = rate * lots * number of rollover crossings.

        A rollover is charged each time the configured rollover time (UTC) is
        crossed between entry and exit. Returns a positive cost magnitude (USD),
        which ``calculate_pnl_with_costs`` subtracts from gross P&L.
        """
        if not EXECUTION.get("enable_swap_fees", False):
            return 0.0
        rate = float(EXECUTION.get("swap_fee_per_lot_per_day", 0.0) or 0.0)
        if rate <= 0.0 or entry_time is None or exit_time is None or position_size <= 0:
            return 0.0
        try:
            hh, mm = (int(x) for x in str(EXECUTION.get("rollover_time_utc", "21:59")).split(":"))
        except Exception:
            hh, mm = 21, 59
        from datetime import timedelta
        try:
            roll = entry_time.replace(hour=hh, minute=mm, second=0, microsecond=0)
        except Exception:
            return 0.0
        if roll <= entry_time:
            roll += timedelta(days=1)
        nights = 0
        while roll <= exit_time:
            nights += 1
            roll += timedelta(days=1)
        return rate * position_size * nights

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

        # Refine the raw SL/TP reason into an honest exit taxonomy. Because
        # disable_tp_when_trailing routes most WINNERS out through a moved stop that
        # execution.check_exit labels "SL", the raw counts conflate profit-protecting
        # trailing/BE exits with true stop-loss losses. Split them:
        #   SL_LOSS    -> original stop hit (a real loss)
        #   BE_STOP    -> stop had been moved to ~breakeven (protected, ~0R)
        #   TRAIL_STOP -> trailing stop had activated (locked-in profit)
        if reason == "SL":
            if trade.trailing_active:
                reason = "TRAIL_STOP"
            elif trade.sl_moved_to_be:
                reason = "BE_STOP"
            else:
                reason = "SL_LOSS"
            trade.exit_reason = reason

        # Calculate MFE/MAE in R-multiples
        stop_distance = abs(trade.entry_price - sl_for_r)
        if stop_distance > 0:
            if trade.direction == "LONG":
                trade.mfe_r = (trade.best_price_in_favor - trade.entry_price) / stop_distance
                trade.mae_r = (trade.entry_price - trade.worst_price_against) / stop_distance
            else:
                trade.mfe_r = (trade.entry_price - trade.best_price_in_favor) / stop_distance
                trade.mae_r = (trade.worst_price_against - trade.entry_price) / stop_distance

        # Use remaining size if partial TP was taken
        position_size_for_exit = trade.remaining_size if trade.partial_tp_taken else trade.position_size

        # Overnight swap cost: charged per rollover crossing between entry and exit.
        swap_cost = self._calculate_swap_cost(trade.entry_time, trade.exit_time, position_size_for_exit)
        trade.swap_cost = swap_cost

        # Calculate gross and net P&L (with costs)
        gross_pnl, net_pnl, total_costs = calculate_pnl_with_costs(
            entry_price=trade.entry_price,
            exit_price=exit_price,
            position_size=position_size_for_exit,
            direction=trade.direction,
            commission=trade.commission_cost * (1 - trade.partial_close_pct) if trade.partial_tp_taken else trade.commission_cost,
            spread_cost=trade.spread_cost * (1 - trade.partial_close_pct) if trade.partial_tp_taken else trade.spread_cost,
            slippage_cost=trade.slippage_cost * (1 - trade.partial_close_pct) if trade.partial_tp_taken else trade.slippage_cost,
            swap_cost=swap_cost,
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

        # Phase 2: update drawdown-control state (loss streak + equity peak)
        if net_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        if self.current_balance > self.equity_peak:
            self.equity_peak = self.current_balance

        # Update cost stats
        self.cost_stats["total_spread_cost"] += trade.spread_cost
        self.cost_stats["total_slippage_cost"] += trade.slippage_cost
        self.cost_stats["total_commission_cost"] += trade.commission_cost
        self.cost_stats["total_swap_cost"] = self.cost_stats.get("total_swap_cost", 0.0) + swap_cost
        self.cost_stats["total_costs"] += trade.total_costs
        self.cost_stats["gross_pnl"] += gross_pnl
        self.cost_stats["net_pnl"] += net_pnl

        # Record exit PnL for session tracking
        ts = candle.get("timestamp")
        self.time_filter.record_trade_exit(ts, net_pnl)

        # Save trade
        self.trades.append(trade)

        # E-adaptive: rolling confidence-bucket recalibration on closed trades only
        self._maybe_recalibrate_confidence()

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

    # -------------------------------------------------------------------------
    # E-adaptive: rolling confidence recalibration (walk-forward safe — only
    # trailing CLOSED trades feed it). Buckets whose trailing win rate falls
    # below the breakeven WR implied by their own realized win/loss sizes get
    # gated; they are re-admitted once enough gate-window time passes that the
    # window would have refreshed. This is the self-optimization loop: the
    # system prunes decaying signal buckets live instead of relying on the
    # in-sample calibration forever.
    # -------------------------------------------------------------------------

    def _maybe_recalibrate_confidence(self):
        from config import CONFIDENCE_SIZING

        cfg = CONFIDENCE_SIZING.get("adaptive", {})
        if not cfg.get("enabled", False):
            return
        every = cfg.get("recalib_every", 20)
        if len(self.trades) == 0 or len(self.trades) % every != 0:
            return

        window = cfg.get("window_trades", 60)
        min_n = cfg.get("min_bucket_n", 10)
        timeout = cfg.get("gate_timeout_trades", 40)

        # A gated bucket produces no new trades, so its trailing stats freeze —
        # re-admit on timeout to re-test it rather than waiting on frozen data.
        for label, gated_at in list(self._adaptive_gated.items()):
            if len(self.trades) - gated_at >= timeout:
                del self._adaptive_gated[label]
                self.adaptive_events.append(
                    {"trade_n": len(self.trades), "action": "READMITTED_TIMEOUT",
                     "bucket": label})

        recent = self.trades[-window:]
        buckets = {}
        for t in recent:
            label = getattr(t, "confidence_label", "") or "UNKNOWN"
            buckets.setdefault(label, []).append(t)

        for label, ts in buckets.items():
            if len(ts) < min_n or label in self._adaptive_gated:
                continue
            wins = [t for t in ts if t.net_pnl > 0]
            losses = [t for t in ts if t.net_pnl <= 0]
            if not losses:
                continue
            wr = len(wins) / len(ts)
            avg_win_r = (sum(abs(t.r_multiple) for t in wins) / len(wins)) if wins else 0.0
            avg_loss_r = sum(abs(t.r_multiple) for t in losses) / len(losses)
            if avg_loss_r <= 0:
                continue
            # Breakeven WR for the bucket's own realized payoff profile
            be_wr = avg_loss_r / (avg_loss_r + avg_win_r) if avg_win_r > 0 else 1.0
            if wr < be_wr:
                self._adaptive_gated[label] = len(self.trades)
                self.adaptive_events.append(
                    {"trade_n": len(self.trades), "action": "GATED", "bucket": label,
                     "trailing_wr": round(wr, 3), "breakeven_wr": round(be_wr, 3)})

    # -------------------------------------------------------------------------
    # Market Chase — enter at close for high-quality missed fills (real trade)
    # -------------------------------------------------------------------------

    def _attempt_market_chase(self, entry, risk, candle, current_idx, df, atr,
                              confidence_tier, tier_risk_pct, pattern_key, verbose):
        """Attempt to enter at candle close when a high-quality limit order misses.

        Returns True if a real trade was created, False otherwise.
        """
        from config import RISK_MODEL

        self.chase_stats["chase_attempts"] += 1

        # Gate 1: Direction filter
        direction_filter = MARKET_CHASE.get("direction_filter", "LONG")
        if direction_filter and direction_filter != "BOTH" and entry.direction != direction_filter:
            self.chase_stats["chase_rejected_direction"] += 1
            return False

        # Gate 2: Confluence score
        min_score = MARKET_CHASE.get("min_confluence_score", 5)
        confluence_score = getattr(entry, "confluence_score", 0)
        if confluence_score < min_score:
            self.chase_stats["chase_rejected_confluence"] += 1
            return False

        chase_entry_price = candle["close"]
        sl_price = risk.stop_loss

        # Gate 3: Stop validity — close must not be past SL
        if entry.direction == "LONG":
            stop_distance = chase_entry_price - sl_price
        else:
            stop_distance = sl_price - chase_entry_price

        if stop_distance <= 0:
            self.chase_stats["chase_rejected_risk_invalid"] += 1
            return False

        # Gate 4: Slippage cap — close must not be too far from limit level
        original_retest_level = getattr(entry, "desired_entry_price", entry.entry_price)
        max_slip_atr = MARKET_CHASE.get("max_entry_slippage_atr", 1.5)
        if entry.direction == "LONG":
            entry_slip = chase_entry_price - original_retest_level
        else:
            entry_slip = original_retest_level - chase_entry_price
        if entry_slip > max_slip_atr * atr:
            self.chase_stats["chase_rejected_slippage"] += 1
            return False

        # All gates passed — recalculate risk for worse entry
        contract_size = RISK_MODEL["contract_size"]
        reward_distance = stop_distance * self.min_rr
        if entry.direction == "LONG":
            tp_price = chase_entry_price + reward_distance
        else:
            tp_price = chase_entry_price - reward_distance

        risk_amount = self.current_balance * tier_risk_pct
        risk_per_lot = stop_distance * contract_size
        lots = risk_amount / risk_per_lot
        lots = round(lots / RISK_MODEL["lot_step"]) * RISK_MODEL["lot_step"]
        lots = max(RISK_MODEL["min_lot"], min(lots, RISK_MODEL["max_lot"]))

        # Build RiskParams for the chase entry
        chase_risk = RiskParams(
            valid=True,
            stop_loss=sl_price,
            take_profit=tp_price,
            stop_distance=stop_distance,
            reward_distance=reward_distance,
            risk_reward_ratio=self.min_rr,
            position_size=lots,
            risk_amount_usd=risk_amount,
            risk_per_lot_usd=risk_per_lot,
            notional_usd=chase_entry_price * contract_size * lots,
        )

        # Build costs via execution engine
        spread_cost = self.execution._calculate_spread_cost(lots)
        slippage_cost = self.execution._calculate_slippage_cost(lots, atr)
        commission_cost = self.execution._calculate_commission(lots)

        chase_fill = FillResult(
            filled=True,
            fill_price=chase_entry_price,
            fill_model="MARKET_CHASE",
            fill_reason=f"Chase entry: confluence {confluence_score}, slip {entry_slip:.2f}",
            costs=CostBreakdown(
                spread_cost=spread_cost,
                slippage_cost=slippage_cost,
                commission_cost=commission_cost,
            ),
        )

        # Execute as a real trade — same path as normal fill
        self.state.seen_patterns.add(pattern_key)

        if entry.fvg_confluence:
            self.confluence_stats["entries_with_fvg"] += 1
        if entry.ob_confluence:
            self.confluence_stats["entries_with_ob"] += 1
        if entry.bos_confirmed:
            self.confluence_stats["entries_with_bos"] += 1

        self.rejection_stats["entries_executed"] += 1
        self.chase_stats["chase_executed"] += 1
        self._pending_confidence_tier = confidence_tier
        self._pending_risk_pct = tier_risk_pct
        self._pending_entry_model = "AMD"
        self._pending_exit_style = ""
        self._pending_sweep_kind = ""
        self._execute_entry(entry, chase_risk, chase_fill, candle, current_idx, df, verbose)

        ts = candle.get("timestamp")
        self.time_filter.record_trade(ts)

        if verbose:
            logger.info(
                f"CHASE ENTRY: {entry.direction} @ {chase_entry_price:.2f} "
                f"(limit was {original_retest_level:.2f}, slip ${entry_slip:.2f}) | "
                f"Confluence: {confluence_score} | Lots: {lots:.2f}"
            )

        return True

    # -------------------------------------------------------------------------
    # Phantom Fills — simulate missed limit orders at candle close
    # -------------------------------------------------------------------------

    def _capture_phantom_trade(self, entry, risk, candle, current_idx, df, atr,
                               confidence_tier, tier_risk_pct, verbose):
        """Simulate a missed fill trade entering at candle close."""
        from config import RISK_MODEL

        phantom_entry_price = candle["close"]
        original_retest_level = getattr(entry, "desired_entry_price", entry.entry_price)
        sl_price = risk.stop_loss

        # Entry slippage: how much worse is close vs limit?
        if entry.direction == "LONG":
            entry_slippage = phantom_entry_price - original_retest_level
            stop_distance = phantom_entry_price - sl_price
        else:
            entry_slippage = original_retest_level - phantom_entry_price
            stop_distance = sl_price - phantom_entry_price

        if stop_distance <= 0:
            return  # Close is past the stop — invalid

        # Recalculate TP with same R:R
        reward_distance = stop_distance * self.min_rr
        if entry.direction == "LONG":
            tp_price = phantom_entry_price + reward_distance
        else:
            tp_price = phantom_entry_price - reward_distance

        # Recalculate position size for wider stop
        contract_size = RISK_MODEL["contract_size"]
        risk_amount = self.current_balance * tier_risk_pct
        risk_per_lot = stop_distance * contract_size
        lots = risk_amount / risk_per_lot
        lots = round(lots / RISK_MODEL["lot_step"]) * RISK_MODEL["lot_step"]
        lots = max(RISK_MODEL["min_lot"], min(lots, RISK_MODEL["max_lot"]))

        # Forward-scan to simulate exit
        exit_result = self._simulate_phantom_exit(
            direction=entry.direction,
            entry_price=phantom_entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            original_sl=sl_price,
            position_size=lots,
            entry_idx=current_idx,
            df=df,
        )
        if exit_result is None:
            return

        phantom = {
            "entry_time": str(candle.get("timestamp", "")),
            "exit_time": exit_result["exit_time"],
            "direction": entry.direction,
            "entry_price": round(phantom_entry_price, 2),
            "exit_price": exit_result["exit_price"],
            "sl_price": round(sl_price, 2),
            "tp_price": round(tp_price, 2),
            "position_size": lots,
            "r_multiple": exit_result["r_multiple"],
            "exit_reason": exit_result["exit_reason"],
            "gross_pnl": exit_result["gross_pnl"],
            "net_pnl": exit_result["net_pnl"],
            "original_retest_level": round(original_retest_level, 2),
            "entry_slippage": round(entry_slippage, 2),
            "bars_to_resolution": exit_result["bars_to_resolution"],
            "confluence_score": getattr(entry, "confluence_score", 0),
            "confidence_tier": confidence_tier,
            "bos_confirmed": getattr(entry, "bos_confirmed", False),
            "fvg_confluence": getattr(entry, "fvg_confluence", False),
            "ob_confluence": getattr(entry, "ob_confluence", False),
            "be_activated": exit_result.get("be_activated", False),
            "trailing_activated": exit_result.get("trailing_activated", False),
        }
        self.phantom_trades.append(phantom)

        if verbose:
            outcome = "WIN" if exit_result["r_multiple"] > 0 else "LOSS"
            logger.info(
                f"PHANTOM ({outcome}): {entry.direction} @ {phantom_entry_price:.2f} "
                f"(limit was {original_retest_level:.2f}, slip ${entry_slippage:.2f}) | "
                f"Exit: {exit_result['exit_reason']} | R: {exit_result['r_multiple']:.2f} | "
                f"Net: ${exit_result['net_pnl']:.2f}"
            )

    def _simulate_phantom_exit(self, direction, entry_price, sl_price, tp_price,
                               original_sl, position_size, entry_idx, df):
        """Forward-scan candles to find phantom trade exit. Stateless — no engine mutation."""
        from config import RISK_MODEL

        max_bars = min(self.max_trade_duration, len(df) - entry_idx - 1)
        if max_bars <= 0:
            return None

        current_sl = sl_price
        best_price = entry_price
        trailing_active = False
        be_activated = False
        contract_size = RISK_MODEL["contract_size"]
        be_trigger_r = STRATEGY.get("move_sl_to_be_at_r", 0)
        trailing_enabled = STRATEGY.get("trailing_stop_enabled", False)
        disable_tp_trailing = STRATEGY.get("disable_tp_when_trailing", False)
        stop_distance_ref = entry_price - original_sl if direction == "LONG" else original_sl - entry_price

        for offset in range(1, max_bars + 1):
            idx = entry_idx + offset
            if idx >= len(self._highs):
                break

            h = float(self._highs[idx])
            l = float(self._lows[idx])
            atr = float(self._atrs[idx])

            # Causality: exits are checked against the stop/TP as they stood
            # at bar start; BE/trail staged from this bar apply from the next
            # bar (same contract as _check_exit and the research lab).
            sl_at_bar_start = current_sl
            tp_disabled_at_bar_start = disable_tp_trailing and trailing_active

            # Update best price in favor (stages BE/trail for the NEXT bar)
            if direction == "LONG":
                best_price = max(best_price, h)
            else:
                best_price = min(best_price, l)

            # Apply BE stop
            if be_trigger_r > 0 and not be_activated and stop_distance_ref > 0:
                if direction == "LONG":
                    current_r = (best_price - entry_price) / stop_distance_ref
                else:
                    current_r = (entry_price - best_price) / stop_distance_ref
                if current_r >= be_trigger_r:
                    be_buffer = atr * STRATEGY.get("be_buffer_atr_mult", 0.1)
                    if direction == "LONG":
                        new_be = entry_price + be_buffer
                        if new_be > current_sl:
                            current_sl = new_be
                    else:
                        new_be = entry_price - be_buffer
                        if new_be < current_sl:
                            current_sl = new_be
                    be_activated = True

            # Apply trailing stop
            if trailing_enabled and atr > 0 and stop_distance_ref > 0:
                eff_sl, is_trailing = get_effective_stop_with_trailing(
                    entry_price=entry_price,
                    current_sl=current_sl,
                    current_extreme_price=best_price,
                    stop_distance=stop_distance_ref,
                    atr=atr,
                    direction=direction,
                )
                if is_trailing and eff_sl != current_sl:
                    current_sl = eff_sl
                if is_trailing:
                    trailing_active = True

            # Effective TP (disable when trailing was active at bar start)
            check_tp = tp_price
            if tp_disabled_at_bar_start:
                check_tp = float('inf') if direction == "LONG" else 0.0

            # Check SL/TP hit against start-of-bar stop
            if direction == "LONG":
                sl_hit = l <= sl_at_bar_start
                tp_hit = h >= check_tp
            else:
                sl_hit = h >= sl_at_bar_start
                tp_hit = l <= check_tp

            if sl_hit or tp_hit:
                exit_at_sl = sl_hit if not (sl_hit and tp_hit) else True  # WORST_CASE
                exit_price = sl_at_bar_start if exit_at_sl else check_tp
                exit_reason = "SL" if exit_at_sl else "TP"

                r_multiple = calculate_exit_r_multiple(
                    entry_price, exit_price, original_sl, direction
                )
                price_diff = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
                gross_pnl = price_diff * contract_size * position_size
                costs = self.execution._calculate_commission(position_size)
                net_pnl = gross_pnl - costs

                ts = df.iloc[idx].get("timestamp", "") if idx < len(df) else ""
                return {
                    "exit_price": round(exit_price, 2),
                    "exit_reason": exit_reason,
                    "exit_time": str(ts),
                    "r_multiple": round(r_multiple, 3),
                    "gross_pnl": round(gross_pnl, 2),
                    "net_pnl": round(net_pnl, 2),
                    "bars_to_resolution": offset,
                    "trailing_activated": trailing_active,
                    "be_activated": be_activated,
                }

        # Timeout — exit at last candle close
        last_idx = min(entry_idx + max_bars, len(self._closes) - 1)
        exit_price = float(self._closes[last_idx])
        r_multiple = calculate_exit_r_multiple(entry_price, exit_price, original_sl, direction)
        price_diff = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
        gross_pnl = price_diff * contract_size * position_size
        costs = self.execution._calculate_commission(position_size)
        net_pnl = gross_pnl - costs
        ts = df.iloc[last_idx].get("timestamp", "") if last_idx < len(df) else ""

        return {
            "exit_price": round(exit_price, 2),
            "exit_reason": "TIMEOUT",
            "exit_time": str(ts),
            "r_multiple": round(r_multiple, 3),
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "bars_to_resolution": max_bars,
            "trailing_activated": trailing_active,
            "be_activated": be_activated,
        }

    def _build_phantom_summary(self) -> dict:
        """Build phantom fills analysis summary."""
        if not self.phantom_fills_enabled or not self.phantom_trades:
            return {}

        pdf = pd.DataFrame(self.phantom_trades)
        wins = pdf[pdf["r_multiple"] > 0]
        losses = pdf[pdf["r_multiple"] <= 0]
        total_loss = abs(losses["net_pnl"].sum()) if len(losses) > 0 else 0

        summary = {
            "total_phantom": len(pdf),
            "phantom_wins": len(wins),
            "phantom_losses": len(losses),
            "phantom_win_rate": round(len(wins) / len(pdf) * 100, 2),
            "phantom_avg_r": round(pdf["r_multiple"].mean(), 3),
            "phantom_median_r": round(pdf["r_multiple"].median(), 3),
            "phantom_net_pnl": round(pdf["net_pnl"].sum(), 2),
            "phantom_profit_factor": round(wins["net_pnl"].sum() / total_loss, 2) if total_loss > 0 else float("inf"),
            "avg_entry_slippage": round(pdf["entry_slippage"].mean(), 2),
            "avg_bars_to_resolution": round(pdf["bars_to_resolution"].mean(), 1),
            "by_exit_reason": pdf["exit_reason"].value_counts().to_dict(),
            "by_confluence_score": {},
            "by_direction": {},
            "trades": self.phantom_trades,
        }

        # Breakdown by confluence score
        for score in sorted(pdf["confluence_score"].unique()):
            subset = pdf[pdf["confluence_score"] == score]
            w = len(subset[subset["r_multiple"] > 0])
            l_pnl = abs(subset[subset["net_pnl"] <= 0]["net_pnl"].sum())
            w_pnl = subset[subset["net_pnl"] > 0]["net_pnl"].sum()
            summary["by_confluence_score"][int(score)] = {
                "count": len(subset),
                "wins": w,
                "win_rate": round(w / len(subset) * 100, 1),
                "avg_r": round(subset["r_multiple"].mean(), 3),
                "net_pnl": round(subset["net_pnl"].sum(), 2),
                "profit_factor": round(w_pnl / l_pnl, 2) if l_pnl > 0 else float("inf"),
            }

        # Breakdown by direction
        for d in ["LONG", "SHORT"]:
            subset = pdf[pdf["direction"] == d]
            if len(subset) == 0:
                continue
            w = len(subset[subset["r_multiple"] > 0])
            summary["by_direction"][d] = {
                "count": len(subset),
                "wins": w,
                "win_rate": round(w / len(subset) * 100, 1),
                "avg_r": round(subset["r_multiple"].mean(), 3),
                "net_pnl": round(subset["net_pnl"].sum(), 2),
            }

        return summary

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
        # sl_exits counts TRUE stop-loss losses (legacy "SL" + refined "SL_LOSS").
        sl_exits = exit_reason_counts.get("SL_LOSS", 0) + exit_reason_counts.get("SL", 0)
        be_exits = exit_reason_counts.get("BE_STOP", 0)
        trail_exits = exit_reason_counts.get("TRAIL_STOP", 0)
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

        # MFE/MAE analysis
        mfe_stats = {}
        if "mfe_r" in trades_df.columns:
            avg_mfe = trades_df["mfe_r"].mean()
            median_mfe = trades_df["mfe_r"].median()
            avg_mae = trades_df["mae_r"].mean() if "mae_r" in trades_df.columns else 0
            winners_mfe = trades_df[trades_df["r_multiple"] > 0]["mfe_r"].mean() if wins > 0 else 0
            capture_ratio = (avg_win_r / winners_mfe * 100) if winners_mfe > 0 else 0
            mfe_stats = {
                "avg_mfe_r": round(avg_mfe, 3),
                "median_mfe_r": round(median_mfe, 3),
                "avg_mae_r": round(avg_mae, 3),
                "winners_avg_mfe_r": round(winners_mfe, 3),
                "mfe_capture_pct": round(capture_ratio, 1),
            }
            # MFE by confluence score
            if "confluence_score" in trades_df.columns:
                mfe_by_confluence = {}
                for score_val in sorted(trades_df["confluence_score"].unique()):
                    subset = trades_df[trades_df["confluence_score"] == score_val]
                    mfe_by_confluence[int(score_val)] = {
                        "count": len(subset),
                        "avg_mfe_r": round(subset["mfe_r"].mean(), 3),
                        "avg_r": round(subset["r_multiple"].mean(), 3),
                        "capture_pct": round(
                            (subset[subset["r_multiple"] > 0]["r_multiple"].mean() /
                             subset[subset["r_multiple"] > 0]["mfe_r"].mean() * 100)
                            if len(subset[subset["r_multiple"] > 0]) > 0 and
                               subset[subset["r_multiple"] > 0]["mfe_r"].mean() > 0
                            else 0, 1
                        ),
                    }
                mfe_stats["by_confluence"] = mfe_by_confluence

        # Move potential analysis
        move_potential_stats = {}
        if "move_potential" in trades_df.columns:
            for mp_val in sorted(trades_df["move_potential"].unique()):
                subset = trades_df[trades_df["move_potential"] == mp_val]
                mp_wins = len(subset[subset["r_multiple"] > 0])
                move_potential_stats[int(mp_val)] = {
                    "count": len(subset),
                    "wins": mp_wins,
                    "win_rate": round(mp_wins / len(subset) * 100, 1) if len(subset) > 0 else 0,
                    "avg_r": round(subset["r_multiple"].mean(), 3),
                    "avg_mfe_r": round(subset["mfe_r"].mean(), 3) if "mfe_r" in subset.columns else 0,
                    "pnl": round(subset["net_pnl"].sum(), 2),
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
            "be_exits": be_exits,
            "trail_exits": trail_exits,
            "tp_exits": tp_exits,
            "rollover_exits": rollover_exits,
            "timeout_exits": timeout_exits,
            # E-adaptive recalibration audit trail
            "adaptive_events": list(getattr(self, "adaptive_events", [])),
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
            "mfe_stats": mfe_stats,
            "move_potential_stats": move_potential_stats,
            "phantom_fills": self._build_phantom_summary(),
            "chase_stats": self.chase_stats if self.market_chase_enabled else {},
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
