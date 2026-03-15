"""
AMD Strategy Configuration
All strategy parameters and system settings in one place.

GOLD INTRADAY STRATEGY - Default settings optimized for XAUUSD M5.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (directory containing config.py) so it works regardless of cwd
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)

# =============================================================================
# MetaTrader 5 Configuration
# =============================================================================
MT5_CONFIG = {
    "login": int(os.getenv("MT5_LOGIN", "0")),
    "password": os.getenv("MT5_PASSWORD", ""),
    "server": os.getenv("MT5_SERVER", ""),
    "path": os.getenv("MT5_PATH", ""),  # Path to terminal64.exe if needed
}

# =============================================================================
# Telegram (Live Signal Notifications)
# =============================================================================
TELEGRAM = {
    "enabled": os.getenv("TELEGRAM_ENABLED", "false").lower() == "true",
    "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    "disable_notification": False,  # silent messages
}

# =============================================================================
# Database Configuration
# =============================================================================
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "amd_trading"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

DATABASE_URL = (
    f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
    f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
)

# =============================================================================
# TIME CONFIGURATION
# =============================================================================
# IMPORTANT: MT5 timestamps are naive (no timezone info).
# Set data_timezone to match your broker's server timezone.
# If broker uses UTC, leave as "UTC". If broker uses GMT+2 (e.g., Europe/Athens), set accordingly.
TIME_CONFIG = {
    # Timezone for interpreting MT5 OHLC timestamps
    # Common values: "UTC", "Europe/Athens", "America/New_York"
    "data_timezone": os.getenv("DATA_TIMEZONE", "UTC"),

    # Timezone for session filter evaluation (kill zone, Asian session, etc.)
    # Recommend keeping this as UTC for consistency with strategy rules
    "session_timezone": os.getenv("SESSION_TIMEZONE", "UTC"),
}

# =============================================================================
# EXECUTION MODEL - Realistic Backtesting
# =============================================================================
# Enabled by default - produces more realistic backtest results
EXECUTION = {
    "enabled": True,

    # Spread Model
    # OHLC data from MT5 is typically BID prices
    "assume_ohlc_is_bid": True,
    "spread_model": "FIXED",  # "FIXED" or "COLUMN" (if df has 'spread' column)
    "fixed_spread_points": 0.25,  # $0.25 typical baseline for XAUUSD

    # Commission (round-turn, i.e., open + close)
    "commission_per_lot_round_turn": 7.0,  # USD per lot round-turn

    # Slippage Model
    # Note: slippage values are in USD/oz (price-space), NOT MT5 points.
    # This differs from spread_points which uses MT5 points (1 pt = $0.01).
    "slippage_model": "ATR_MULT",  # "NONE", "FIXED_POINTS", "ATR_MULT"
    "slippage_points": 0.05,  # USD/oz; used if slippage_model = "FIXED_POINTS"
    "slippage_atr_mult": 0.02,  # 2% of ATR (USD/oz); used if slippage_model = "ATR_MULT"

    # Intrabar Fill Ambiguity
    # When both SL and TP are touched in the same candle:
    # - "WORST_CASE": Assume SL hit first (conservative, recommended)
    # - "BEST_CASE": Assume TP hit first (optimistic)
    # - "RANDOM": Randomly choose based on seed
    "intrabar_fill_rule": "WORST_CASE",
    "random_seed": 42,

    # Entry Fill Model
    # - "CLOSE": Fill at candle close (legacy behavior)
    # - "LIMIT_AT_RETEST": Fill at retest level if touched (default)
    # - "LIMIT_AT_FVG_EDGE": Fill at FVG boundary
    # - "LIMIT_AT_OB": Fill at Order Block level
    "entry_fill_model": "LIMIT_AT_RETEST",

    # Exit Fill Model
    "exit_fill_model": "TOUCH",  # Touch-based with bid/ask adjustment

    # Swap Fees (overnight holding costs)
    "enable_swap_fees": True,
    "swap_fee_per_lot_per_day": 0.0,  # Set based on broker; 0 = disabled

    # Rollover time (when swaps are charged)
    # Typically 21:59 or 22:00 UTC for most brokers
    "rollover_time_utc": "21:59",
}

# =============================================================================
# SESSION FILTER - Time-based entry restrictions
# =============================================================================
# Gold intraday strategy: trade during London/NY overlap (Kill Zone)
SESSION_FILTER = {
    "enabled": True,

    # Kill Zone - focus on London/NY sessions
    # Times are in session_timezone (UTC by default)
    "kill_zone_start": "06:00",  # Pre-London (blackout handles worst hours)
    "kill_zone_end": "20:00",    # Extended into late NY

    # Asian Session avoidance - DISABLED to allow Asian session trading
    # Note: Session spans midnight (23:00 -> 08:00)
    "avoid_asian": False,
    "asian_start": "23:00",
    "asian_end": "08:00",

    # Consolidation detection in Asian session
    # Allow pattern DETECTION during Asian (useful for setups that complete in London)
    "allow_consolidation_in_asian": True,
    # If True, REQUIRE consolidation to form during Asian session
    "require_consolidation_in_asian": False,

    # Daily trade limits
    "max_trades_per_day": 3,  # Allow more setups per day

    # Daily loss limit - stop new entries if daily drawdown exceeds this
    "daily_loss_limit_pct": 0.008,  # 0.8% daily loss limit — stops trading earlier on bad days

    # Cooldown after trade (reduce overtrading in chop)
    "cooldown_minutes_after_trade": 15,

    # Blackout hours — block specific UTC hours with proven negative expectancy
    "blackout_hours_utc": [6, 7, 9, 19],  # Block 06-07 UTC (40% WR, -$51), 09:00 UTC (11.1% WR), 19:00 UTC (0% WR)

    # Blackout weekdays — block entire days with proven negative expectancy
    "blackout_weekdays": [],  # Empty = no weekday blackout (Wed filter tested: hurt Nov, net negative)

    # Overnight holding avoidance
    # Close positions before rollover to avoid swap fees
    "close_before_rollover": False,
    "close_before_rollover_minutes": 5,  # Close 5 min before rollover
    "no_new_entries_before_rollover_minutes": 15,  # No new entries 15 min before

    # ==========================================================================
    # Session-based Pattern Timing (SMC Improvement)
    # ==========================================================================
    # Require consolidation to form during Asian session (23:00-08:00 UTC)
    "require_consolidation_in_asian": False,  # Allow accumulation any session
    # Require distribution to occur during London/NY (08:00-16:00 UTC)
    "require_distribution_in_london_ny": False,  # Allow distribution any session

    # Specific Kill Zones for higher-quality entries
    "london_open_kz": ("08:00", "10:00"),
    "ny_open_kz": ("13:00", "15:00"),
    "only_trade_in_kz": False,  # Trade full London/NY session
}

# =============================================================================
# NEWS FILTER - Avoid high-impact USD news
# =============================================================================
# Disabled by default; enable and provide news CSV to use
NEWS_FILTER = {
    "enabled": True,  # Avoid high-impact event volatility
    "csv_path": "Market/data/news_events.csv",

    # Blackout window around news events
    "pre_minutes": 5,   # No entries from T-5 minutes
    "post_minutes": 15,  # Until T+15 minutes

    # Expected CSV columns:
    # - timestamp: datetime in UTC (required)
    # - currency: e.g., "USD" (optional, for filtering)
    # - impact: e.g., "HIGH", "MEDIUM", "LOW" (optional)
    # - title: event name (optional)
    "impact_filter": ["HIGH"],  # Only block HIGH impact events
}

# =============================================================================
# KEY LEVELS - PDH/PDL, Weekly, Monthly levels
# =============================================================================
KEY_LEVELS = {
    "enabled": True,

    # Level types to use
    "use_pdh_pdl": True,       # Previous Day High/Low
    "use_weekly_high_low": True,  # Previous Week High/Low
    "use_monthly_high_low": True,  # Previous Month High/Low

    # How close entry/retest must be to key level (as ATR multiple)
    "tolerance_atr_mult": 0.35,

    # Mode:
    # - "SCORE": Add confluence score points for key level proximity
    # - "REQUIRE": Block entries that don't meet min score
    "mode": "SCORE",  # Add to confluence score

    # Minimum key level score required (only used if mode = "REQUIRE")
    "min_keylevel_score_required": 1,

    # Score weights for each level type
    "score_weights": {
        "pdh_pdl": 1,
        "weekly": 1,
        "monthly": 1,
    },
}

# =============================================================================
# HTF BIAS - Higher Timeframe Trend Filter
# =============================================================================
# Use H4 and D1 trends to filter entries
HTF_BIAS = {
    "enabled": True,

    # Timeframes for bias calculation
    "primary_timeframe": "H4",
    "secondary_timeframe": "D1",

    # Bias detection method
    "method": "EMA_CROSS",  # EMA crossover method
    "ema_fast": 20,
    "ema_slow": 50,

    # Alignment requirements
    "require_primary_alignment": True,   # Entry must align with H4 trend
    "require_secondary_alignment": False,  # D1 as bonus, not requirement

    # What to do when bias is NEUTRAL
    # - "BLOCK": No entries when neutral
    # - "ALLOW": Allow entries even when neutral
    "neutral_policy": "BLOCK",  # Block entries during H4 neutral (ALLOW was no-op for trade count)
}

# =============================================================================
# VOLUME FILTER
# =============================================================================
VOLUME_FILTER = {
    "enabled": True,

    # Volume MA period for baseline
    "volume_ma_period": 20,

    # Distribution candle must have volume >= this ratio vs consolidation avg
    "distribution_volume_ratio_min": 1.3,  # Volume confirmation required

    # Minimum tick volume (block very low liquidity candles)
    "min_tick_volume": 200,
}

# =============================================================================
# FUNDAMENTALS FILTER - DXY and Real Yields
# =============================================================================
# Disabled by default; enable and provide CSVs to use
FUNDAMENTALS = {
    "enabled": False,

    # Data file paths (CSV format)
    "dxy_csv_path": "Market/data/dxy.csv",
    "real_yields_csv_path": "Market/data/real_yields.csv",

    # Resample rule (align to M5)
    "resample_rule": "5min",

    # MA periods for trend detection
    "dxy_ma_period": 50,
    "yields_ma_period": 50,

    # Gold fundamentals logic:
    # - Long Gold: DXY down AND real yields down (typically)
    # - Short Gold: DXY up AND real yields up
    "require_dxy_down_for_long": True,
    "require_yields_down_for_long": True,
    "require_dxy_up_for_short": True,
    "require_yields_up_for_short": True,

    # Safe haven override - bypass fundamentals checks
    # Use during extreme risk-off events when gold rallies regardless
    "safe_haven_override": False,
}

# =============================================================================
# RISK MODEL - Position Sizing and Stop Placement
# =============================================================================
# XAUUSD-correct math using contract size (not pip-based)
RISK_MODEL = {
    # XAUUSD: 1 lot = 100 oz, so $1 price move = $100 per lot
    "contract_size": 100.0,

    # Risk per trade (lower = smaller drawdowns, smoother monthly curve)
    "risk_pct_per_trade_default": 0.003,  # 0.3% base risk (confidence sizing scales up for best setups)
    "risk_pct_per_trade_max": 0.02,       # 2% max allowed

    # Stop loss placement
    # Minimum stop distance as ATR multiple (avoid too-tight stops)
    "min_stop_atr_mult": 1.5,      # Wider stops = fewer noise stop-outs
    # Buffer beyond manipulation extreme
    "stop_buffer_atr_mult": 1.0,   # More buffer to avoid noise hits

    # Leverage/notional guard
    # notional = entry_price * contract_size * lots
    # Guard: notional <= balance * max_position_notional_multiple
    "max_position_notional_multiple": 300.0,  # 1:300 leverage

    # Lot size constraints
    "min_lot": 0.01,
    "max_lot": 50.0,
    "lot_step": 0.01,

    # ==========================================================================
    # Partial Take Profit Settings (SMC Improvement)
    # ==========================================================================
    # Disabled to eliminate 0R cluster from partial TP + BE stop
    "partial_tp_enabled": False,         # Was True - simplified exit
    "partial_tp_at_1r": 0.5,              # Close 50% of position at 1R (unused when disabled)
    "move_sl_to_be_after_partial": True,  # Move SL to breakeven after partial
    "final_tp_rr": 3.0,                   # Let remaining position run to 3R
}

# =============================================================================
# Strategy Parameters - AMD (Accumulation/Manipulation/Distribution)
# Optimized for XAU/USD M5 - more trades, better expectancy
# =============================================================================
STRATEGY = {
    # Symbol and timeframe
    "symbol": "XAUUSD",
    "timeframe": "M5",

    # Phase 1: Consolidation - tight ranges only (genuine accumulation)
    "consolidation_lookback": 12,
    "consolidation_range_atr_mult": 4.00,   # Accept slightly wider consolidations
    "consolidation_close_pct": 0.60,         # Accept 60% closes inside range

    # Phase 2: Manipulation - decisive stop hunts only
    "manipulation_break_atr_mult": 0.20,     # Must break meaningfully beyond range
    "manipulation_return_candles": 12,       # Allow slower manipulation reversals

    # Phase 3: Distribution - strong conviction breakout
    "distribution_break_atr_mult": 0.20,    # Real move must be significant
    "distribution_body_mult": 1.30,         # Slightly lower body expansion bar
    "distribution_follow_through_candles": 2,  # Reduced follow-through requirement
    "distribution_require_extension": False,   # Don't require new extreme

    # Phase 4: Entry - quality at entry point
    "rejection_wick_ratio": 0.20,          # Accept more wick patterns at retest
    "retest_tolerance_atr_mult": 0.55,      # Wider retest zone — biggest lever for trade count

    # Phase 5: Risk Management (uses RISK_MODEL now, kept for compatibility)
    "min_rr": 2.0,                          # 2:1 is still good R:R
    "max_risk_pct": 0.01,                   # 1% per trade
    "spread_buffer_pips": 10,               # Legacy; use RISK_MODEL stop_buffer_atr_mult

    # Directional Filter - both directions with quality filters
    "allow_short_trades": True,             # Both longs and shorts enabled

    # ATR Settings
    "atr_period": 14,

    # ==========================================================================
    # SMC Confluence Settings (Fair Value Gap, Order Block, Break of Structure)
    # ==========================================================================

    # Entry Mode: "RETEST_ONLY", "RETEST_WITH_FVG", "ORDER_BLOCK", "PEAK_LOW"
    "entry_mode": "RETEST_ONLY",            # No FVG required - more trades

    # Fair Value Gap (FVG) Settings
    "fvg_min_size_atr_mult": 0.10,          # Min FVG size as ATR multiple
    "fvg_lookback": 20,                     # Candles to search for FVG

    # Order Block Settings
    "ob_min_body_atr_mult": 0.15,           # Min OB body size as ATR multiple
    "ob_displacement_mult": 1.5,            # Required move after OB (body multiple)

    # Break of Structure (BOS) Settings
    "bos_swing_lookback": 5,
    "bos_required": True,                   # Require BOS confirmation (W-only)

    # ==========================================================================
    # SMC Improvements - Quality Filters and Refinements
    # ==========================================================================

    # FVG Quality Scoring - Filter weak FVGs
    "fvg_min_quality_score": 1,             # Min quality score (0-3) to use FVG, 0=disabled
    "fvg_quality_large_mult": 0.3,          # Size > this ATR mult = +1 score
    "fvg_quality_xlarge_mult": 0.5,         # Size > this ATR mult = +1 more score
    "fvg_impulse_body_ratio": 0.7,          # Impulse candle body ratio > this = +1 score

    # FVG Entry Style - Where to enter relative to FVG
    # "equilibrium": 50% of FVG (midpoint) - better R:R
    # "optimal": Best price (furthest into FVG)
    # "exit": Edge of FVG (conservative)
    "fvg_entry_style": "equilibrium",

    # Order Block Body Only - Use candle body instead of full range
    "ob_use_body_only": True,               # True = body (open/close), False = range (high/low)

    # Liquidity Sweep Confirmation
    "require_liquidity_sweep": False,       # Use as confluence bonus, not hard gate
    "liquidity_sweep_lookback": 50,

    # Volume spike during manipulation (stop hunt creates volume)
    "require_manipulation_volume_spike": False,  # Use as confluence bonus, not hard gate
    "manipulation_volume_spike_ratio": 1.5,

    # Equal highs/lows (liquidity pools) within consolidation
    "detect_equal_levels": True,
    "equal_level_tolerance_atr_mult": 0.05,
    "equal_level_min_touches": 2,
    "prefer_equal_level_sweep": True,      # Prefer setups that sweep equal highs/lows

    # Breaker Blocks - Broken OBs become breakers for opposite direction
    "use_breaker_blocks": True,             # Was False - enable breaker block confluence

    # Minimum confluence score required for entry (BOS + FVG + OB + equal levels + volume + breaker)
    "min_confluence_score": 3,              # Require 3+ factors — score-2 had 61.5% loss rate

    # Premium/Discount Zones - off to allow more entries
    "require_discount_for_long": False,
    "require_premium_for_short": False,

    # Trailing stop - let winners breathe
    "trailing_stop_enabled": True,
    "trailing_stop_activation_r": 2.0,      # Baseline value — trail bug is fixed so it'll work when reached
    "trailing_stop_atr_mult": 2.5,          # Baseline value — wide trail lets big winners run

    # Breakeven stop - protect at 1R
    "move_sl_to_be_at_r": 1.0,             # Move SL to breakeven at 1.0R profit (proven: 65.7% activation rate)

    # Short trade quality gate (shorts need higher confluence)
    "short_min_confluence_score": 3,

    # Judas Swing quality filters
    "judas_max_candles": 5,                 # Fast sweeps (1-5 candles)
    "judas_min_velocity_atr": 0.3,          # Decisive sweeps
    "judas_london_bonus": True,             # Extra confluence for London session manipulation
    "judas_london_start_hour": 7,           # London open window start (UTC)
    "judas_london_end_hour": 10,            # London open window end (UTC, inclusive)

    # Order Block detection
    "ob_lookback": 10,                      # How far back to search for OB candle

    # Judas quality hard gate - use as confluence bonus, not hard kill
    "min_judas_quality": 0,                 # Disabled as hard gate; scored in confluence

    # Consolidation quality gate
    "min_consolidation_quality": 0,         # Removed quality gate

    # Stale retest filter
    "max_bars_after_distribution": 40,      # Allow later retests (widened from 30)

    # Disable TP when trailing stop is active — let trail manage the exit
    "disable_tp_when_trailing": True,

    # Max trade duration in bars (M5 candles)
    "max_trade_duration": 240,  # 240 bars = 20h on M5 (compromise: less MTM DD than 300, more room than 200)
}

# =============================================================================
# Backtesting Configuration
# =============================================================================
BACKTEST = {
    "initial_capital": 100.0,
    "commission_per_lot": 7.0,              # Commission in USD per lot (legacy)
    "pip_value": 0.01,                      # XAU pip value (legacy, prefer RISK_MODEL)
    "lot_size": 100,                        # 1 lot = 100 oz for XAU (legacy)
}

# =============================================================================
# Validation Targets
# =============================================================================
# =============================================================================
# CONFIDENCE-BASED POSITION SIZING
# =============================================================================
# Instead of flat risk on every trade, tier risk by trade quality.
# High-confidence setups (score 4+ in prime hours) get more risk,
# amplifying winners while keeping base risk low.
CONFIDENCE_SIZING = {
    "enabled": True,

    # Base risk (applied to all trades that don't match any tier)
    "base_risk_pct": 0.003,        # 0.3% — conservative base

    # Tier definitions: checked top-down, first match wins
    "tiers": [
        {
            "name": "high",
            "min_confluence_score": 5,
            "prime_hours_only": True,     # H13-17 UTC
            "risk_pct": 0.008,            # 0.8% — elite setups only
        },
        {
            "name": "standard",
            "min_confluence_score": 3,
            "prime_hours_only": True,
            "risk_pct": 0.005,            # 0.5% — prime hours, score 3+
        },
        {
            "name": "off_hours",
            "min_confluence_score": 5,
            "prime_hours_only": False,
            "risk_pct": 0.005,            # 0.5% — high-confluence off-hours (not punished)
        },
        # Fallthrough: base_risk_pct (0.3%) for everything else
        # MEDIUM tier removed — 48.4% loss rate, -$245 damage
    ],

    # Prime hours definition (UTC)
    "prime_hours_start": 13,
    "prime_hours_end": 17,    # inclusive: H13, H14, H15, H16, H17
}

# =============================================================================
# Phantom Fills Analysis
# =============================================================================
# When enabled, simulates trades that passed all filters but whose limit entry
# was never filled ("Fill Not Triggered"). Enters at candle close instead.
# Results tracked separately — never affects real trades or equity.
PHANTOM_FILLS = {
    "enabled": False,               # Must explicitly enable via --phantom-fills
    "entry_price_mode": "CLOSE",    # Enter at candle close when limit misses
    "recalculate_risk": True,       # Recalculate position size for worse entry
}

# =============================================================================
# Market Chase — Enter at close when limit misses on high-quality setups
# =============================================================================
MARKET_CHASE = {
    "enabled": False,               # Must explicitly enable via --market-chase
    "direction_filter": "LONG",     # Only chase LONGs (60.2% WR vs 50% SHORT)
    "min_confluence_score": 5,      # Score 5: 67.6% WR, PF 2.12
    "max_entry_slippage_atr": 1.5,  # Reject if close is > 1.5 ATR worse than limit
}

# =============================================================================
# Validation Targets
# =============================================================================
VALIDATION = {
    "min_trades": 30,                      # Fewer but higher quality trades
    "min_expectancy_r": 0.25,               # Slightly higher bar
    "max_drawdown_pct": 0.20,               # Max drawdown < 20%
    "min_months": 3,                        # Minimum months of data
}


# =============================================================================
# Configuration Validation
# =============================================================================
def validate_config() -> list[str]:
    """Check for conflicting or problematic configuration settings.

    Returns a list of warning messages.  Empty list means no issues found.
    """
    warnings = []

    # Max possible confluence score (BOS + FVG + OB + equal_levels + volume + breaker = 6)
    max_possible = 6
    min_req = STRATEGY.get("min_confluence_score", 0)
    if min_req > max_possible:
        warnings.append(
            f"min_confluence_score ({min_req}) exceeds max possible ({max_possible}). "
            "No entries will ever pass."
        )
    short_min = STRATEGY.get("short_min_confluence_score", min_req)
    if short_min > max_possible:
        warnings.append(
            f"short_min_confluence_score ({short_min}) exceeds max possible ({max_possible})."
        )

    # Risk bounds
    risk_pct = RISK_MODEL.get("risk_pct_per_trade_default", 0)
    risk_max = RISK_MODEL.get("risk_pct_per_trade_max", 0)
    if risk_pct > risk_max:
        warnings.append(
            f"risk_pct_per_trade_default ({risk_pct}) exceeds risk_pct_per_trade_max ({risk_max})."
        )
    if risk_pct <= 0:
        warnings.append("risk_pct_per_trade_default is zero or negative — no trades will size.")

    # Confidence tiers risk vs max risk
    if CONFIDENCE_SIZING.get("enabled"):
        for tier in CONFIDENCE_SIZING.get("tiers", []):
            tier_risk = tier.get("risk_pct", 0)
            if tier_risk > risk_max:
                warnings.append(
                    f"Confidence tier '{tier.get('name')}' risk ({tier_risk}) "
                    f"exceeds risk_pct_per_trade_max ({risk_max})."
                )

    # Timezone validity
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(TIME_CONFIG.get("data_timezone", "UTC"))
        ZoneInfo(TIME_CONFIG.get("session_timezone", "UTC"))
    except Exception as e:
        warnings.append(f"Invalid timezone in TIME_CONFIG: {e}")

    # Trailing stop activation should be reachable
    trailing_r = STRATEGY.get("trailing_stop_activation_r", 0)
    be_r = STRATEGY.get("move_sl_to_be_at_r", 0)
    if STRATEGY.get("trailing_stop_enabled") and trailing_r > 0 and be_r >= trailing_r:
        warnings.append(
            f"BE stop at {be_r}R activates at/after trailing at {trailing_r}R — "
            "trailing may never activate."
        )

    return warnings