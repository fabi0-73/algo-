"""
AMD Strategy Configuration
All strategy parameters and system settings in one place.

GOLD INTRADAY STRATEGY - Default settings optimized for XAUUSD M5.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (directory containing config.py) so it works regardless of cwd
_PROJECT_ROOT = Path(__file__).resolve().parent
_env_path = _PROJECT_ROOT / ".env"
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
    # Timezone for interpreting MT5 OHLC timestamps.
    # IMPORTANT (verified empirically): MT5 (via src/data/mt5_client.py) stores bar
    # times as the BROKER SERVER wall-clock (IC Markets ≈ GMT+2/+3, i.e. New York +7h;
    # the daily data gap at hour 00 == 17:00 New York settlement confirms this). We keep
    # this labeled "UTC" on purpose: the backtest AND the live scanner both read candles
    # through the same MT5Client with no conversion, so they share one consistent frame.
    # => All SESSION_FILTER/blackout/kill-zone hours below are in BROKER time, not true
    #    UTC. US news events must likewise be expressed in broker time (NY+7): NFP 08:30
    #    ET -> 15:30 here, FOMC 14:00 ET -> 21:00 here (DST-proof, constant +7 from NY).
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
    # Value is in MT5 POINTS (1 pt = $0.01/oz). Cost = points * 0.01 * contract_size(100) * lots.
    # Realistic XAUUSD spread ~25-30 cents/oz => ~$25-30 per 1.0 lot.
    # (The old value 0.25 modeled only $0.0025/oz ≈ $0.25/lot — ~100x too tight, which
    #  materially overstated backtest P&L. 30 matches the code's own fallback default.)
    "fixed_spread_points": 30.0,  # ~30 cents/oz spread (crossed once at entry)

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

    # Swap Fees (overnight holding costs) — now actually applied in engine._exit_position.
    # Cost magnitude in USD per 1.0 lot per night held (positive = you pay). Charged once
    # per rollover crossing between entry and exit. PLACEHOLDER — replace with your broker's
    # actual XAUUSD swap (gold swaps are typically negative, i.e. a cost).
    "enable_swap_fees": True,
    "swap_fee_per_lot_per_day": 5.0,

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
    # (require_consolidation_in_asian lives in the Session-based Pattern
    # Timing section below — it was duplicated here and the later key won)
    "allow_consolidation_in_asian": True,

    # Daily trade limits
    "max_trades_per_day": 3,  # Allow more setups per day

    # Daily loss limit - stop new entries if daily drawdown exceeds this
    "daily_loss_limit_pct": 0.008,  # 0.8% daily loss limit — stops trading earlier on bad days

    # Monthly loss limit - stop new entries for the rest of the month once monthly loss
    # exceeds this. Activates the previously-dormant circuit breaker in time_filters.py
    # (the key was missing from config, so it defaulted to 0.0 = disabled).
    # 5% was tested (run 70c1d5f1) and hurt: 415 blocked entries vs 141, deeper DD
    # (blocked recovery). 6% is the validated value.
    "monthly_loss_limit_pct": 0.06,  # 6% monthly loss limit

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
    # Absolute path from project root. (Was "Market/data/news_events.csv" — a
    # non-existent nested Market/ dir — which made the filter silently self-disable.)
    # Event timestamps in this CSV must be in BROKER time (NY+7) to match the candle
    # frame; generate with: python scripts/generate_news_events.py
    "csv_path": str(_PROJECT_ROOT / "data" / "news_events.csv"),

    # If True, an ENABLED filter with a missing/unreadable CSV raises instead of
    # silently disabling — so a run never trades "unprotected" without you knowing.
    "require_csv": True,

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

    # Data file paths (CSV format) — absolute from project root (see NEWS_FILTER note).
    "dxy_csv_path": str(_PROJECT_ROOT / "data" / "dxy.csv"),
    "real_yields_csv_path": str(_PROJECT_ROOT / "data" / "real_yields.csv"),

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

    # Leverage/notional guard (a hard ceiling, NOT the primary risk control).
    # notional = entry_price * contract_size * lots; guard: notional <= balance * mult.
    # Actual per-trade risk is governed by risk_pct + the Phase-2 drawdown controls;
    # with 0.3-0.8% risk and the min_lot floor, real leverage used sits far below this.
    # Kept generous per the "$500, allow higher leverage" choice — lower to 100.0 to
    # hard-cap at 1:100. (Higher leverage on a small account raises risk of ruin.)
    "max_position_notional_multiple": 300.0,

    # Lot size constraints
    # 0.10 floor was tested at user request (run 75542d59): account DESTROYED — 5 trades,
    # final capital -$287 (negative), one wide-stop trade lost $620, breaker latched for
    # the entire test (19,266 blocked entries). At $500, min_lot 0.10 risks 25-120% per
    # trade. Reverted to the validated 0.01 (run 124d15ef). Aggressive-but-survivable
    # option: 0.02 (bootstrap: 1.9% ruin, ~2x trajectory). 0.10 becomes sane ~$5,000 equity.
    "min_lot": 0.01,
    "max_lot": 50.0,
    "lot_step": 0.01,

    # ==========================================================================
    # Partial Take Profit Settings (SMC Improvement)
    # ==========================================================================
    # RE-TESTED 2026-07-02 with honest costs (train run a8fd25ca) and REJECTED again:
    # WR jumped 39%->55% but expectancy fell 0.248R->0.152R, PF 1.47->1.32, max DD
    # 19.9%->24.1%, net -38%. Banking half at 1R halves the trailing runners that pay
    # for the drawdowns. Win rate here is cosmetic; do not re-enable to chase it.
    "partial_tp_enabled": False,
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

    # --- previously-hidden knobs, surfaced 2026-07-10 (defaults unchanged) ---
    # Pattern-search window: how many bars after a consolidation ends the
    # scanner looks for manipulation/distribution. Bounds trade count.
    "pattern_min_bars_after_consolidation": 10,
    "pattern_max_bars_after_consolidation": 60,
    # Scan stride over candidate consolidation windows (engine + live share
    # it). 2 = evaluate every second window; 1 doubles candidates AND compute.
    # NEVER change without the full validation battery — more candidates
    # changes which patterns dedup first.
    "consolidation_scan_step": 2,
    # Breakeven cushion above/below entry when the BE stop moves (ATR mult).
    "be_buffer_atr_mult": 0.1,
    # Entry order type at the retest ("LIMIT" validated; "MARKET" = chase,
    # tested and rejected in run c19e123e) + rejection-candle fallback for
    # FVG mode (never validated on).
    "entry_execution": "LIMIT",
    "allow_rejection_fallback": False,

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
    "prefer_equal_level_sweep": True,      # UNUSED — no consumers in code (kept for compat; see SWEEP_MODEL for the real sweep logic)

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

    # E3 OTE refinement: "RETEST" places the entry limit at the consolidation edge
    # (validated behavior); "OTE" moves it to the 61.8-79% deep-pullback band of the
    # manipulation->distribution leg when that is a strictly better price.
    # EXPERIMENT KILLED 2026-07-04 (train run bqtakw5ls): 2,353 limits placed,
    # ZERO filled in 12.6 months. A 61.8% retrace of the manip->distribution leg
    # sits inside the consolidation range — reaching it invalidates the AMD
    # pattern itself. Deep pullbacks and valid retests are structurally mutually
    # exclusive here; the consolidation edge IS the optimal fillable entry.
    # E2 (OB+OTE standalone) skipped per pre-registered condition. KEEP "RETEST".
    "entry_price_mode": "RETEST",

    # E3 companion: count "retest level in OTE band" as a +1 confluence factor.
    # OFF by default — enabling shifts scores/tiers on the validated champion.
    "use_ote_confluence": False,

    # E4 overnight-drift overlay: LONG trades with an ACTIVE trailing stop may run
    # up to this many bars instead of max_trade_duration (0 = disabled). Basis:
    # gold overnight returns are positive at peer-reviewed significance
    # (J Econ Finance 2017); risk is trailing-protected and swap is modeled.
    # EXPERIMENT RESULT 2026-07-04 (run 695b6ece vs 53d5bfc4): NO-OP — results
    # byte-identical to baseline. The 2-ATR trail always exits runners before the
    # 240-bar timeout, so the extension never engages. Harvesting overnight drift
    # would require loosening the validated trail itself — not attempted. KEEP 0.
    "long_runner_max_duration": 0,
}

# =============================================================================
# Backtesting Configuration
# =============================================================================
BACKTEST = {
    "initial_capital": 500.0,
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
            "lot_bonus": 0.02,            # +0.02 lots on top of calculated size
        },
        {
            "name": "standard",
            "min_confluence_score": 3,
            "prime_hours_only": True,
            "risk_pct": 0.005,            # 0.5% — prime hours, score 3+
            "lot_bonus": 0.01,            # +0.01 lots on top of calculated size
        },
        {
            "name": "off_hours",
            "min_confluence_score": 5,
            "prime_hours_only": False,
            "risk_pct": 0.005,            # 0.5% — high-confluence off-hours (not punished)
            "lot_bonus": 0.01,            # +0.01 lots on top of calculated size
        },
        # Fallthrough: base_risk_pct (0.3%) for everything else
    ],

    # Prime hours definition (UTC)
    "prime_hours_start": 13,
    "prime_hours_end": 17,    # inclusive: H13, H14, H15, H16, H17

    # E-adaptive: rolling self-recalibration of confidence buckets. Every
    # recalib_every closed trades, each bucket's trailing WR (last window_trades)
    # is compared to the breakeven WR implied by its own realized win/loss sizes;
    # below-breakeven buckets stop trading and are re-tested after
    # gate_timeout_trades. Walk-forward safe: only closed trades feed it.
    # EXPERIMENT RESULT 2026-07-04 (run 991eeaf5 vs champion 124d15ef): the
    # mechanism WORKED (gated LOW at trade 60 when trailing WR hit 15%, re-tested
    # at 100) and raised WR 39.4->44.5% with expectancy nearly intact (0.391R vs
    # 0.402R) — the only WR lever that didn't wreck expectancy. But final capital
    # $2,044 vs $2,125 and DD 23.6% vs 20.2% -> does not beat champion. KEEP
    # DISABLED; a near-neutral option if higher WR ever matters more than P&L.
    "adaptive": {
        "enabled": False,
        "window_trades": 60,
        "recalib_every": 20,
        "min_bucket_n": 10,
        "gate_timeout_trades": 40,
    },
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
# VALIDATION FAILED 2026-07-02 (run c19e123e): despite phantom trades showing LONG
# misses at +0.369R, REAL chase trades collapsed the system — 142 -> 76 total trades,
# expectancy 0.402R -> 0.097R, DD 25.7%, equity hit the 20% circuit breaker and halted.
# Cause: phantoms ignore position exclusivity (chase trades displace better organic
# setups) and enter at worse prices with min-lot-pinned dollar risk. KEEP DISABLED.
MARKET_CHASE = {
    "enabled": False,               # Failed validation — see note above
    "direction_filter": "LONG",
    "min_confluence_score": 3,
    "max_entry_slippage_atr": 1.5,
}

# =============================================================================
# Adaptive Exits (stub — not yet implemented)
# =============================================================================
ADAPTIVE_EXITS = {
    "enabled": False,
    "tiers": [],
    "extend_duration_for_runners": False,
    "runner_max_duration": 300,
}

# =============================================================================
# DRAWDOWN CONTROLS (Phase 2) — the core "lower drawdown" machinery
# =============================================================================
# These run INSIDE the backtest engine (previously such controls existed only in
# src/live/monitor.py and were never wired into backtests). Three layers:
#   1) Account circuit breaker — halt NEW entries when equity drawdown from peak is
#      severe, resume after it recovers (hysteresis). Mirrors the live kill switch.
#   2) Consecutive-loss brake — cut risk after a losing streak until the next win.
#   3) Equity-based risk scaling — smoothly de-risk the deeper the drawdown, so the
#      equity curve is smoother and bad runs shrink position size automatically.
# Toggle the whole system with "enabled" (or CLI --no-drawdown-controls).
#
# RE-TUNED 2026-07-09 to catastrophe-only after the virgin-window inversion:
# the shipped 20% breaker + scaling turned a +$447 window (run be08230a raw)
# into -$113 (run f3cc3fa4 shipped) by halting through the recovery, and cost
# ~$300 over the full 23mo arc (no-controls f997b260 +$2,036 / AMD-only
# f754d882 +$2,363 vs ~+$1,730 shipped composite) to cap a DD that unhalted
# peaked at 24.55% and recovered to new highs. On a min-lot account the stack
# cannot size down, so halting was its only lever — and halting amputates the
# trailing-runner recoveries that fund the system. New shape: breaker at 30%
# (pure tail insurance — never fired in any measured arc), scaling and
# consec-loss brake OFF (both were on in the losing config, off in both
# winners), monthly 6% loss limit KEPT (active in the winning runs).
DRAWDOWN_CONTROLS = {
    "enabled": True,

    # 1) Account-level circuit breaker (peak->current realized-equity drawdown)
    "circuit_breaker_enabled": True,
    "max_account_dd_pct": 0.30,      # catastrophe-only: above any measured arc (24.55%)
    "resume_dd_pct": 0.15,           # resume once DD recovers to <= 15% (hysteresis)
    # Deadlock fix: with entries halted and no open position, equity can never recover,
    # so a pure DD-recovery resume would latch forever (seen in run c19e123e: 7,503
    # blocked entries). After this many bars halted, resume anyway — the risk-scaling
    # floor keeps size halved while still in deep drawdown. Mirrors the live monitor's
    # "manual review then resume smaller". 1440 bars = ~5 trading days on M5.
    "halt_max_bars": 1440,

    # 2) Consecutive-loss brake — OFF 2026-07-09 (on in the losing shipped
    # config, off in both winning arcs f997b260/f754d882)
    "consec_loss_enabled": False,
    "max_consecutive_losses": 4,     # after 4 losses in a row...
    "consec_loss_risk_factor": 0.5,  # ...trade at 50% risk until the next win

    # 3) Equity-based (drawdown) risk scaling — OFF 2026-07-09: impotent at the
    # min-lot floor, proven twice (tightening test 70c1d5f1 made DD worse;
    # full-arc f997b260 shows the whole scaling+halt stack subtracts value).
    "risk_scaling_enabled": False,
    "risk_scale_start_dd": 0.06,     # start scaling risk down once DD > 6%
    "risk_scale_full_dd": 0.15,      # reach the floor factor at 15% DD
    "risk_scale_min_factor": 0.5,    # never size below 50% of the tier's base risk
    "suppress_lot_bonus_when_scaling": True,  # don't add fixed lot_bonus while de-risking
}

# =============================================================================
# SIGNAL CONFIDENCE — empirical per-trade confidence + optional HIGH up-sizing
# =============================================================================
# Score/label come from entry.calculate_signal_confidence (calibrated on 280
# honest-cost trades; GOOD/HIGH buckets verified on the latest 30% window).
# Every trade gets the label (backtest record + live Telegram signal).
# Sizing: on a $500 account positions are pinned to the 0.01 min-lot floor, so
# risk-% multipliers do nothing — the effective lever is extra lots on the best
# bucket. Sequence simulation of "HIGH +0.01 lot" on run f99ef66e: +300%→+353%
# return, est. max DD 17%→22%. Needs full-backtest validation (controls interact).
SIGNAL_CONFIDENCE = {
    "enabled": True,                 # compute + display confidence on every trade/signal
    "size_by_confidence": True,      # up-size HIGH-confidence trades (moderate policy)
    "high_extra_lots": 0.01,         # extra lots on HIGH (score 4) — 0.01→0.02 on $500
    "good_extra_lots": 0.0,          # extra lots on GOOD (score 3) — off by default (DD cost)
    # Entry gate: skip trades with confidence score below this (0 = disabled).
    # E-conf-gate EXPERIMENT KILLED 2026-07-04 (train runs 53d5bfc4 vs 5d5091be):
    # blocking LOW (scores 0-1) cut trades 102->81 but WR FELL 39.2->38.3%, DD
    # worsened 20.2->23.6%, final capital dropped $892->$864. Expectancy gain
    # (0.248->0.265R) doesn't pay for lost recovery trades. Confidence is a
    # SIZING signal, not an entry gate — consistent with every prior "fewer
    # trades" experiment on this account size. KEEP 0.
    "min_confidence_to_trade": 0,

}

# =============================================================================
# SWEEP MODEL — liquidity-sweep reversal entries ("liquidation point" strategy)
# =============================================================================
# Second entry model alongside AMD: fade a sweep of an OHLCV-derived stop-cluster
# level (equal highs/lows, PDH/PDL, prior-week H/L, Asian range, $-round numbers).
# Evidence basis & caveats in src/strategy/liquidity_levels.py / sweep_entry.py.
#
# EXPERIMENT OUTCOME (2026-07-02, train split Sep24-Sep25, honest costs):
#   FIXED_RR 0.030R PF 0.99 | HYBRID 0.084R PF 0.90 | LONG-only+HYBRID 0.132R PF 1.09
#   All FAIL the kill bar (train expectancy >= 0.15R) -> stays DISABLED (mode "AMD").
#   The edge exists but is too thin after realistic spread/slippage — exactly what the
#   Turtle-Soup literature predicted. Positive pockets for future work (do NOT cherry-
#   pick without fresh OOS): Asian-low sweeps LONG (+0.6-0.7R both styles, n=14),
#   prime hours 13-17. Runs: 3095645a / 44ae42aa / bfb9e9e0.
SWEEP_MODEL = {
    "enabled": True,
    # "AMD" = current behavior only; "SWEEP" = sweep entries only; "BOTH" = combined.
    "strategy_mode": "AMD",          # default unchanged until validation passes

    # --- Liquidity level sources (see liquidity_levels.get_active_levels) ---
    "use_pdh_pdl": True,
    "use_weekly": True,
    "use_asian_range": True,
    "use_round_numbers": True,
    "round_step": 25.0,              # gold psychological increments ($25)
    "use_equal_levels": True,
    "equal_tolerance_atr_mult": 0.10,
    "equal_min_touches": 2,
    "level_lookback": 200,           # bars of swing history for equal levels
    "swing_strength": 3,

    # --- Sweep trigger ---
    # LONG-only: fade sell-side sweeps (below lows) only. SHORT fades were negative in
    # BOTH exit styles on the train split (-0.47R/-0.55R) AND in the phantom-fill data
    # (-0.13R, n=275) — consistent with gold's long-side bias in this period.
    "long_only": True,
    "min_poke_atr_mult": 0.10,       # wick must exceed level by >= 0.10 ATR
    "max_candles_back_inside": 3,    # close back inside within 3 bars of the poke
    "require_rejection": True,       # trigger bar must reject (wick-through or counter body)
    "max_level_distance_atr": 1.5,   # trigger close must be within 1.5 ATR of the level
    "volume_bonus": True,            # soft flag only (tick volume is a proxy) — never gates
    "volume_bonus_ratio": 1.5,

    # --- Exits (A/B under test; per-trade via TradeRecord.exit_style) ---
    # "FIXED_RR": TP at tp_rr, BE at STRATEGY move_sl_to_be_at_r, NO trailing.
    # "HYBRID":   close hybrid_partial_pct at hybrid_partial_at_r, BE, trail rest.
    "exit_style": "FIXED_RR",
    "tp_rr": 2.5,
    "hybrid_partial_at_r": 1.0,
    "hybrid_partial_pct": 0.5,
}

# =============================================================================
# NY Initial Balance pullback model (second live stream candidate)
# =============================================================================
# Lab-validated (runs 65fecd3b/6ef3f2fb, src/research/strategies/ny_ib.py):
# train 103tr 72.8% WR +0.057R PF 1.45 | OOS 71tr 83.1% WR +0.121R PF 2.15,
# robust to $0.45/oz spread. IB = first NY hour range (16:30-17:30 broker,
# = 9:30-10:30 ET); after a close beyond the IB within the scan window, a
# LIMIT order rests back inside the range; small TP beyond the edge, wide SL
# across the range (gold usually breaks only ONE side of its IB per day).
# ENGINE-VALIDATED & ENABLED (2026-07-06):
#   NY_IB-only  run 19af4776: 157tr 77.1% WR +0.096R PF 1.78 DD 7.6%
#   NY_IB OOS   run 12a01408:  66tr 78.8% WR +0.108R PF 1.82 DD 9.0%
#   BOTH (AMD+NY_IB) run 3f3ea9d1: 256tr 60.2% WR +0.266R PF 2.06 DD 19.5%,
#     final $2,342 vs champion 124d15ef $2,125 — MORE profit, LOWER DD, +80%
#     trades. Passes the pre-registered accept bar (beats champion, DD<25%).
# PAUSED 2026-07-09 pending re-validation: the regime broke Mar-Jul 2026
#   (77-83% WR -> ~46% in the virgin window, run be08230a) and the full-arc
#   no-controls comparison shows NY_IB SUBTRACTS by position displacement —
#   AMD-only f754d882 (+$2,363, DD 22.5%, PF 2.00, Sharpe 3.29) beats
#   BOTH f997b260 (+$2,036, DD 24.6%, PF 1.72) on EVERY metric; NY_IB's own
#   141 trades netted ~+$17. Re-enable only after a fresh lab pass on recent
#   data (scripts/mtf_lab.py --strategies ny_ib) clears the pre-registered bar.
NY_IB_MODEL = {
    "enabled": False,                # PAUSED 2026-07-09 — see note above
    "only": False,                   # True = suppress AMD (validation runs)

    "ib_start": "16:30",             # broker time (= 09:30 ET, DST-locked)
    "ib_end": "17:30",
    "scan_end": "22:00",             # breakout close must occur before this
    "eod_flat": "23:00",             # force-flat and pending-order expiry
    "min_ib_bars": 10,               # of 12 possible M5 bars in the IB hour
    "ib_min_pct": 0.004,             # IB size as fraction of price
    "ib_max_pct": 0.02,

    "retrace_frac": 0.10,            # LIMIT this far back inside the range
    "sl_range_mult": 1.00,           # SL 1.0x IB from entry (the wide stop IS
    "tp_range_mult": 0.20,           #   the design; RR ~0.2 with ~73-83% WR)
    "max_trades_day": 1,
    "long_only": False,              # both sides validated

    # Pure bracket: no BE, no trailing, no partial — engine exit_style BRACKET.
    "exit_style": "BRACKET",
    # Its 17:30-22:00 window sits outside AMD killzones; skip ONLY the
    # killzone/asian session gate. News blackout, loss limits, cooldown,
    # rollover and the drawdown breaker still apply.
    "bypass_session_window": True,
    # The lab-validated spec has no HTF-bias gate (it is an AMD-frame concept);
    # with it, engine run 0f778191 rejected 82 of 172 breakouts and halved the
    # stream's frequency. Bypass to match the validated design.
    "bypass_htf_bias": True,
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
# RUN TIMEFRAME — run the whole pipeline on resampled higher-TF bars
# =============================================================================
# Philosophy: pattern parameters stay in BARS on purpose — a 12-bar consolidation
# on M15 is a 3-hour accumulation, i.e. a structurally BIGGER setup than on M5,
# which is the whole point of a higher-TF run. Only wall-clock plumbing scales:
#   - halt_max_bars (circuit-breaker deadlock timeout ~= 5 trading days)
#   - minute-width windows (news pre-blackout, rollover no-entry window) are
#     floored to one bar interval so they can't fall between bars.
# M5 remains the default and is completely unchanged by this layer.
RUN_TIMEFRAME = {
    "timeframe": "M5",
    "bar_minutes": 5,
}

_HALT_MAX_BARS_M5 = 1440  # canonical M5 value; scaled per timeframe below


def apply_run_timeframe(timeframe: str) -> None:
    """Mutate config dicts for a resampled higher-TF run. Call before engine init."""
    from src.data.resample import TIMEFRAME_MINUTES  # local import: avoid cycle

    minutes = TIMEFRAME_MINUTES[timeframe]
    RUN_TIMEFRAME["timeframe"] = timeframe
    RUN_TIMEFRAME["bar_minutes"] = minutes
    if minutes == 5:
        return

    # Keep the deadlock timeout at ~5 trading days of wall-clock
    DRAWDOWN_CONTROLS["halt_max_bars"] = max(1, int(_HALT_MAX_BARS_M5 * 5 / minutes))

    # Floor sub-bar windows to one bar so they cannot fall between bar timestamps
    NEWS_FILTER["pre_minutes"] = max(NEWS_FILTER.get("pre_minutes", 5), minutes)
    NEWS_FILTER["post_minutes"] = max(NEWS_FILTER.get("post_minutes", 15), minutes)
    SESSION_FILTER["no_new_entries_before_rollover_minutes"] = max(
        SESSION_FILTER.get("no_new_entries_before_rollover_minutes", 15), minutes)
    SESSION_FILTER["close_before_rollover_minutes"] = max(
        SESSION_FILTER.get("close_before_rollover_minutes", 5), minutes)
    SESSION_FILTER["cooldown_minutes_after_trade"] = max(
        SESSION_FILTER.get("cooldown_minutes_after_trade", 15), minutes)


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