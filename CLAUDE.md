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

# Persistent 24/7 scanner (Windows Scheduled Task; survives session/reboot)
powershell -ExecutionPolicy Bypass -File scripts/install_live_task.ps1  # install (once)
# then: Start-ScheduledTask AMD_Live_Scanner | Stop-ScheduledTask AMD_Live_Scanner
# status: Get-ScheduledTask AMD_Live_Scanner | Get-ScheduledTaskInfo
# log:    logs/live_scanner.log  (supervised loop restarts the scanner on crash;
#         needs the user logged on — MT5 is a GUI terminal)

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

Trades require `min_confluence_score` (default 3) from these factors: BOS confirmed (+1), FVG at retest (+1), Order Block at retest (+1), equal level swept (+1), volume spike (+1), breaker block (+1), Judas quality ≥2 (+1). FVG/OB are always searched regardless of entry mode to feed this score.

### Execution Model

Realistic cost modeling: spread (fixed or column-based), slippage (ATR-multiplied), commission ($7/lot round-turn). Intrabar ambiguity: WORST_CASE assumes SL hit first when both SL and TP are touched in same candle. Entry fills use LIMIT_AT_RETEST model.

### Risk & Position Sizing

Uses contract size of 100 oz/lot (XAUUSD). Position size = Risk Amount / (Entry - SL) / Contract Size. Default risk: 0.3% per trade (confidence tiers up-size; on small accounts the 0.01 min-lot floor dominates and `risk_amount_usd` reports the FLOORED dollar risk). Stops placed at manipulation extreme + buffer (1.0 ATR), minimum 1.5 ATR distance. Breakeven stop activates at 1R (cushion `be_buffer_atr_mult`), trailing stop at 2.0R with 2.5 ATR trail. Stop moves computed from a bar take effect on the NEXT bar (causality contract, tests/test_causality.py).

### Quality Gates (in order within `_scan_for_patterns`)

1. Consolidation quality score (`score_consolidation_quality`, min 1) — checks range tightness, close%, equal levels, duration
2. Judas quality (`min_judas_quality`, default 0 = disabled as hard gate) — fast sweep, velocity, London timing
3. Liquidity sweep / volume spike — can be required or used as confluence bonus
4. Distribution follow-through — requires 1 candle making a new extreme beyond break price (2->1 adopted 2026-07-23, run 439e2edd)
5. BOS required — break of structure must confirm direction
6. Stale retest filter — max 60 bars after distribution (`max_bars_after_distribution`, 40->60 adopted 2026-07-23)

### Backtest Output

Reports saved to `reports/backtest_{id}/` with equity curves, R-distribution, monthly P&L, cost breakdown PNGs, plus report.txt and results.json.

## Data-First Research Pipeline

Statistical research layer in `src/research/` for finding small, frequently-repeating edges (thousands of occurrences) before any strategy is written. Separate from the AMD engine — the engine is never used for research.

### Workflow (in order; each stage gates the next)

```bash
# 1. Fetch the deepest history the broker serves (MT5 here retains only
#    ~15-17 months of M5 — a 2021 --start would yield only what exists)
python scripts/fetch_data.py --months 18

# 2. Export DB -> portable cache (research runs anywhere from this file)
python scripts/export_cache.py --gzip        # data/lab_m5_cache.csv.gz

# 3. Audit the cache (gaps, duplicates, OHLC sanity, coverage) — fix before proceeding
python scripts/audit_data.py --cache data/lab_m5_cache.csv.gz

# 4. Event study: standalone edge of each AMD atom (train window only)
python scripts/event_study.py --events all
python scripts/event_study.py --oos          # SINGLE look at OOS

# 5. Conditional mining: event x context grid, TRAIN ONLY, guardrails in code
python scripts/mine_patterns.py

# 6. Promote survivors by hand to src/research/strategies/<name>.py, then the
#    existing gauntlet: mtf_lab.py -> --oos once -> monte_carlo.py
python scripts/mtf_lab.py --strategies <name>
```

### Key Research Modules

- `src/research/events.py` — 41 vectorized per-bar event detectors (sweeps, FVGs, BOS, OB retests, Judas, session opens, PDH/PDL, VWAP stretch; 2026-07-10 batch: ORB break/pullback, sweep-reclaim, failed-break fade, wick rejection, round-number rejects, vol dry-up, inside-NR7, settlement gaps, PM-fix window, news-reopen, H1 sweeps; 2026-07-23 batch: ratio_stretch gold/silver z-fade, asia_range_ebreak, ema_pullback_reclaim, ribbon_expansion — all four judged NULL on this feed, kept for context mining). Contract: `detect_<name>(df, params) -> [fired, direction, strength]` aligned to df; event knowable at bar-i close. `HORIZONS = (1,3,6,12,24,48)` is fixed upfront — never add horizons per-event. Pure time-of-day atoms are nulled by construction in the TOD-matched excess; their informative stat is the CI-vs-zero.
- `src/research/forward.py` — forward returns/MFE/MAE from `open[i+1]` in ATR units; `cost_in_atr()` is the effect-size floor (spread+commission+slippage ≈ 0.10–0.17 ATR on real gold M5).
- `src/research/stats.py` — numpy-only: day-block bootstrap CI, time-of-day-matched permutation p (normal-tail extension beyond permutation resolution), BH-FDR, `decluster()`.
- `src/research/context.py` — categorical conditioning features (session, dow, ATR regime, H1 trend, PD levels, VWAP side), ≤5 levels each, lookahead-safe.
- `src/research/audit.py` — cache data-quality audit (`audit_m5`).
- `src/research/study.py` — shared cell evaluation for the two study CLIs.

### Research Discipline (enforced in code, don't relax casually)

- Mining is TRAIN-ONLY (`mine_patterns.py` never loads OOS bars). OOS is one look, at promotion time.
- Candidates must pass ALL of: BH-FDR (q=0.10) over the full grid, n ≥ 300 after horizon-declustering, mean net-R ≥ +0.05 after costs, gross excess ≥ 1.5× per-event cost, same-sign mean in both train halves.
- Events are compared against a time-of-day-matched baseline, never the all-day mean.
- New detectors must pass the prefix-invariance test in `tests/test_events.py` (mechanical no-lookahead check) — add planted-pattern tests for the specific geometry.

## Environment Setup

Requires `.env` file (see `env.example.txt`) with MT5 credentials and PostgreSQL connection details.

## Important Tuning Notes

- Parameters interact multiplicatively — tightening multiple gates simultaneously can reduce trades to zero. When tuning, change one category at a time and check the rejection funnel stats.
- The rejection funnel in backtest output shows where trades are filtered at each stage. Custom rejection stats (e.g., `low_consolidation_quality`, `no_judas_quality`) use `setdefault` and may not appear in the standard funnel display.
- Full test suite passes (one legitimate skip: `test_lab_costs.py` parity check when the champion report isn't in the checkout). `tests/test_strategy.py` needs `data/news_events.csv` — generate with `python scripts/generate_news_events.py` on a fresh clone.
