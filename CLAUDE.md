# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AMD Strategy Backtester + Live Signal Scanner for XAU/USD (Gold) on M5 timeframe. Implements the AMD (Accumulation/Manipulation/Distribution) institutional trading strategy with SMC (Smart Money Concepts) confluence, realistic execution modeling, and Monte Carlo validation.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch historical data from MT5
python scripts/fetch_data.py --months 6

# Run backtest (full filters)
python scripts/run_backtest.py --verbose

# Run backtest with specific filters disabled
python scripts/run_backtest.py --no-session --no-news --no-htf --no-keylevel --no-volume

# Run backtest with date range
python scripts/run_backtest.py --start 2024-01-01 --end 2024-06-30

# Run tests
pytest tests/ -v
pytest tests/test_strategy.py::TestRisk -v

# Monte Carlo validation
python scripts/monte_carlo.py --report reports/backtest_<id>

# Walk-forward test
python scripts/walk_forward.py --split 0.7 --monte-carlo

# Live signal scanner (signal-only, no execution)
python scripts/run_live.py

# Database setup
python scripts/setup_database.py
```

## Architecture

### Signal Pipeline (per candle in backtest loop)

```
Candle → detect_consolidation() → detect_manipulation() → detect_distribution()
      → check_entry_at_candle() → [Filters] → calculate_risk() → ExecutionEngine
      → TradeRecord → calculate_metrics()
```

The pipeline scans backwards from each candle looking for completed AMD patterns. Each phase must pass before the next is checked.

### Key Modules

- `config.py` — Master configuration. All strategy parameters, filter settings, execution model, and risk model are defined here as dicts (`STRATEGY`, `EXECUTION`, `SESSION_FILTER`, `KEY_LEVELS`, `HTF_BIAS`, `VOLUME_FILTER`, `RISK_MODEL`, `VALIDATION`).
- `src/strategy/` — AMD phase detectors (consolidation, manipulation, distribution, entry, risk) and SMC modules (fvg, order_blocks, market_structure, indicators). Also contains filter engines (time_filters, news_filter, htf_bias, key_levels, volume_filters, fundamentals).
- `src/backtest/engine.py` — Main backtest loop (~1600 lines). Orchestrates pattern scanning (`_scan_for_patterns`), filter application (`_check_filters`), exit management (`_check_exit`), and trade recording.
- `src/backtest/execution.py` — Fill simulation with spread, slippage, and commission modeling. Handles intrabar ambiguity (WORST_CASE/BEST_CASE/RANDOM).
- `src/backtest/metrics.py` — Performance stats: win rate, expectancy, drawdown, profit factor, cost breakdown.
- `src/data/` — mt5_client.py (MetaTrader 5 connection), db.py (PostgreSQL via SQLAlchemy).

### Filter Pattern

Each filter engine follows a consistent interface: takes DataFrame + config + current state, returns `(bool, reason)`. All filters can be toggled via config or CLI flags (`--no-session`, `--no-news`, `--no-htf`, `--no-keylevel`, `--no-volume`, `--enable-fundamentals`).

### Confluence Scoring

Trades require `min_confluence_score` (default 2) from these factors: BOS confirmed (+1), FVG at retest (+1), Order Block at retest (+1), equal level swept (+1), volume spike (+1), breaker block (+1), Judas quality ≥2 (+1). FVG/OB are always searched regardless of entry mode to feed this score.

### Execution Model

Realistic cost modeling: spread (fixed or column-based), slippage (ATR-multiplied), commission ($7/lot round-turn). Intrabar ambiguity: WORST_CASE assumes SL hit first when both SL and TP are touched in same candle. Entry fills use LIMIT_AT_RETEST model.

### Risk & Position Sizing

Uses contract size of 100 oz/lot (XAUUSD). Position size = Risk Amount / (Entry - SL) / Contract Size. Default risk: 0.5% per trade. Stops placed at manipulation extreme + buffer (1.0 ATR), minimum 1.5 ATR distance. Breakeven stop activates at 1R, trailing stop at 1.5R with 2.0 ATR trail.

### Quality Gates (in order within `_scan_for_patterns`)

1. Consolidation quality score (`score_consolidation_quality`, min 1) — checks range tightness, close%, equal levels, duration
2. Judas quality (`min_judas_quality`, default 0 = disabled as hard gate) — fast sweep, velocity, London timing
3. Liquidity sweep / volume spike — can be required or used as confluence bonus
4. Distribution follow-through — requires 3 candles making new extremes beyond break price
5. BOS required — break of structure must confirm direction
6. Stale retest filter — max 15 bars after distribution

### Backtest Output

Reports saved to `reports/backtest_{id}/` with equity curves, R-distribution, monthly P&L, cost breakdown PNGs, plus report.txt and results.json.

## Environment Setup

Requires `.env` file (see `env.example.txt`) with MT5 credentials and PostgreSQL connection details.

## Important Tuning Notes

- Parameters interact multiplicatively — tightening multiple gates simultaneously can reduce trades to zero. When tuning, change one category at a time and check the rejection funnel stats.
- The rejection funnel in backtest output shows where trades are filtered at each stage. Custom rejection stats (e.g., `low_consolidation_quality`, `no_judas_quality`) use `setdefault` and may not appear in the standard funnel display.
- One pre-existing test failure: `TestDistributionFollowThrough::test_distribution_follow_through_rejection` fails because it calls `_scan_for_patterns` directly without `run()` which initializes `_atrs`.
