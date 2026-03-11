# AMD Strategy Backtester + Live Signal Scanner

Automated backtesting engine and live signal scanner for the **AMD (Accumulation/Manipulation/Distribution)** trading strategy on XAU/USD (Gold) with realistic execution modeling, Judas swing quality filtering, Monte Carlo validation, and SMC confluence.

## Strategy Overview

The AMD strategy identifies institutional price manipulation patterns:

1. **Consolidation (Accumulation)**: Price moves in a tight range, liquidity builds
2. **Manipulation (Fake Breakout)**: Price breaks out, sweeps stops, reverses quickly
3. **Distribution (Real Move)**: Price breaks opposite direction with momentum
4. **Entry**: Enter on retest of the broken level with rejection confirmation

### SMC Confluence Features

- **Fair Value Gaps (FVG)**: Identify imbalanced price zones for entry refinement
- **Order Blocks (OB)**: Detect institutional supply/demand zones
- **Break of Structure (BOS)**: Confirm trend direction changes

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your values:

```bash
# Windows
copy .env.example .env

# Linux / Mac
cp .env.example .env
```

Edit `.env` with your credentials:

| Variable | Description |
|----------|-------------|
| `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` | MetaTrader 5 (live data); set to 0/empty to skip MT5 |
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | PostgreSQL (optional fallback for candles) |
| `DATA_TIMEZONE`, `SESSION_TIMEZONE` | Broker timezone (default: UTC) |
| `TELEGRAM_ENABLED`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Telegram bot for live signal alerts |

### 3. Create Database

Create a PostgreSQL database named `amd_trading`:

```sql
CREATE DATABASE amd_trading;
```

### 4. Fetch Historical Data

```bash
python scripts/fetch_data.py --months 6
```

### 5. Run Backtest

```bash
# Basic backtest
python scripts/run_backtest.py --verbose

# $100 objective profile (300-500 trades + net positive target)
python scripts/run_backtest.py --no-charts --no-session --no-htf

# Same profile with full chart artifacts
python scripts/run_backtest.py --no-session --no-htf

# With date range
python scripts/run_backtest.py --start 2024-01-01 --end 2024-06-30

# Disable filters (raw signals)
python scripts/run_backtest.py --no-session --no-news --no-htf --no-keylevel --no-volume

# Full realism with fundamentals
python scripts/run_backtest.py --enable-fundamentals --intrabar WORST_CASE
```

### 6. Analyze Results

```bash
python scripts/analyze_results.py
```

### 7. Monte Carlo Validation

```bash
# Run Monte Carlo on a completed backtest (10K simulations)
python scripts/monte_carlo.py --report reports/backtest_aae991eb

# With charts and save results
python scripts/monte_carlo.py --report reports/backtest_aae991eb --charts --save

# Custom simulation count
python scripts/monte_carlo.py --report reports/backtest_aae991eb --simulations 20000
```

### 8. Walk-Forward Validation

```bash
# 70/30 train/test split with auto Monte Carlo on test set
python scripts/walk_forward.py --split 0.7 --monte-carlo

# Custom split
python scripts/walk_forward.py --split 0.6
```

### 9. Live Signal Scanner

```bash
# Signal-only mode (default) -- scans and prints signals, no execution
python scripts/run_live.py

# Single scan, no loop
python scripts/run_live.py --once

# Custom balance and scan interval
python scripts/run_live.py --balance 500 --interval 60

# Log signals to file
python scripts/run_live.py --output signals.jsonl

# Send signals to Telegram (requires TELEGRAM_* in .env)
python scripts/run_live.py --telegram

# Test Telegram connection (sends one message and exits)
python scripts/run_live.py --telegram-test
```

**Testing the Telegram node:** Ensure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set in `.env`, then run `python scripts/run_live.py --telegram-test`. You should receive a single message in the chat.

**"Chat not found" (400):** The bot can only send to chats it knows. (1) In Telegram, open your bot and send `/start` (or tap Start). (2) Get your chat ID: open `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates` in a browser (paste your token in place of `<YOUR_BOT_TOKEN>`). In the JSON, find `"chat":{"id": 123456789}` — that number is `TELEGRAM_CHAT_ID` for a private chat. For a group, add the bot to the group, send any message there, then call getUpdates again; use the negative `chat.id` (e.g. `-1001234567890`) in `.env`. Restart the script after changing `.env`.

## Project Structure

```
market/
├── config.py                 # Strategy parameters and settings
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
├── env.example.txt          # Environment variable template
├── src/
│   ├── data/
│   │   ├── mt5_client.py     # MetaTrader 5 connection
│   │   └── db.py             # PostgreSQL database layer
│   ├── strategy/
│   │   ├── indicators.py     # ATR, body size calculations
│   │   ├── consolidation.py  # Phase 1: Range detection
│   │   ├── manipulation.py   # Phase 2: Fake breakout + Judas swing quality
│   │   ├── distribution.py   # Phase 3: Real breakout confirmation
│   │   ├── entry.py          # Phase 4: Retest entry logic + short quality gate
│   │   ├── risk.py           # Phase 5: Risk management (contract-size math)
│   │   ├── fvg.py            # Fair Value Gap detection
│   │   ├── order_blocks.py   # Order Block detection
│   │   ├── market_structure.py  # Swing points and BOS
│   │   ├── time_filters.py   # Kill zone and session filters
│   │   ├── news_filter.py    # News blackout filter
│   │   ├── key_levels.py     # PDH/PDL, weekly, monthly levels
│   │   ├── htf_bias.py       # Higher timeframe bias alignment
│   │   ├── volume_filters.py # Volume confirmation
│   │   └── fundamentals.py   # DXY and real yields filter
│   ├── backtest/
│   │   ├── engine.py         # Main backtesting loop
│   │   ├── execution.py      # Realistic execution model
│   │   └── metrics.py        # Performance calculations
│   ├── live/
│   │   ├── signals.py        # Real-time AMD pattern scanner
│   │   └── monitor.py        # Kill switch and risk monitoring
│   └── visualization/
│       └── charts.py         # Equity curves, Monte Carlo charts
├── scripts/
│   ├── fetch_data.py         # Download historical data
│   ├── run_backtest.py       # Execute backtest with CLI options
│   ├── run_live.py           # Live signal scanner (signal-only mode)
│   ├── monte_carlo.py        # Monte Carlo ruin simulation
│   ├── walk_forward.py       # Walk-forward validation
│   ├── analyze_results.py    # Generate reports
│   ├── debug/                # Debug utilities
│   └── optimization/         # Parameter optimization tools
├── data/
│   ├── news_events.example.csv  # Sample news events file
│   ├── dxy.example.csv          # Sample DXY data
│   └── real_yields.example.csv  # Sample real yields data
└── tests/
    └── test_strategy.py      # Unit tests
```

## Strategy Parameters

All parameters are configurable in `config.py`:

### Core AMD Parameters

| Phase | Parameter | Default | Description |
|-------|-----------|---------|-------------|
| Consolidation | `consolidation_lookback` | 20 | Candles to analyze |
| Consolidation | `consolidation_range_atr_mult` | 0.35 | Max range as ATR multiple |
| Consolidation | `consolidation_close_pct` | 0.70 | Min % closes inside range |
| Manipulation | `manipulation_break_atr_mult` | 0.50 | Min break distance |
| Manipulation | `manipulation_return_candles` | 3 | Max candles to return |
| Distribution | `distribution_break_atr_mult` | 0.30 | Min break for confirmation |
| Distribution | `distribution_body_mult` | 1.50 | Body expansion required |

### Risk Management (Gold-Specific)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_risk_pct` | 0.01 | Max 1% per trade |
| `min_rr` | 2.0 | Minimum risk:reward |
| `leverage_limit` | 100 | Max leverage |
| `contract_size` | 100 | 100 oz per lot (XAUUSD) |
| `max_position_lots` | 10.0 | Maximum position size |

### Execution Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fill_model` | LIMIT_AT_RETEST | Entry fill simulation |
| `intrabar_assumption` | WORST_CASE | SL/TP ambiguity handling |
| `spread_points` | 7 | Bid/ask spread (IC Markets RAW avg) |
| `slippage_model` | ATR_MULT | Slippage calculation |
| `commission_per_lot` | 7.0 | Round-trip commission |

### Session Filters

| Filter | Default | Description |
|--------|---------|-------------|
| Kill Zone | 07:00-10:00 & 13:00-16:00 UTC | London open + NY overlap only |
| Asian Avoidance | 23:00-08:00 UTC | Entries gated by KZ; avoid Asian |
| Max Trades/Day | 5 | Daily trade limit |
| News Blackout | T-5 to T+15 min | Avoid high-impact USD news |
| Rollover | 21:00 UTC | Close positions before rollover |

### HTF Bias & Key Levels

| Feature | Default | Description |
|---------|---------|-------------|
| HTF Primary | H4 | Primary trend timeframe |
| HTF Secondary | D1 | Secondary confirmation |
| Bias Method | EMA cross | 20/50 EMA alignment |
| Key Levels | PDH/PDL, Weekly, Monthly | Confluence scoring |

## CLI Options

### Execution Model Flags

```bash
--fill-model CLOSE|LIMIT_AT_RETEST    # Entry fill simulation
--intrabar WORST_CASE|BEST_CASE|RANDOM # SL/TP ambiguity
--spread 30                            # Spread in points
--slippage NONE|FIXED|ATR_MULT         # Slippage model
--commission 7.0                       # Per-lot commission
```

### Filter Toggles

```bash
--no-session          # Disable kill zone filter
--no-news             # Disable news blackout
--no-htf              # Disable HTF bias filter
--no-keylevel         # Disable key level filter
--no-volume           # Disable volume filter
--enable-fundamentals # Enable DXY/yields filter (off by default)
```

### Output Controls

```bash
--no-charts   # Skip PNG chart rendering (report text/json still saved)
--no-report   # Skip writing report artifacts to reports/
```

## Realistic Execution Model

The backtester includes a sophisticated execution model to avoid curve-fitting:

### Intrabar Ambiguity Handling

When a candle touches both SL and TP, the `intrabar_assumption` setting determines the outcome:

- **WORST_CASE** (default): Assumes SL hit first - most conservative
- **BEST_CASE**: Assumes TP hit first - optimistic
- **RANDOM**: 50/50 random selection

### Cost Modeling

All P&L calculations include:
- **Spread**: Entry cost based on bid/ask spread
- **Slippage**: ATR-based adverse fill
- **Commission**: Per-lot round-trip fee

### Limit Fill Simulation

With `LIMIT_AT_RETEST` fill model, entries only execute if price retests the desired entry level, avoiding optimistic market-order assumptions.

## Data Files

### Optional External Data

For full filter functionality, provide these CSV files in `data/`:

1. **news_events.csv** - High-impact USD news times
2. **dxy.csv** - US Dollar Index daily data
3. **real_yields.csv** - US 10-year real yields

See `data/*.example.csv` for format templates.

## Validation Targets

Before live trading, the strategy must demonstrate:

- **200+ trades** minimum sample size
- **Expectancy > 0.25R** average profit per trade
- **Max drawdown < 20%** of account
- **Positive net P&L** after all costs
- **Walk-forward degradation ratio >= 0.6** (test expectancy / train expectancy)
- **Monte Carlo P95 drawdown < 30%** (worst 5% of simulations)

## Reports

After running a backtest, reports are saved to `reports/backtest_{id}/`:

- `equity_curve.png` - Realized + MTM equity with drawdown
- `r_distribution.png` - Histogram of R-multiples
- `monthly_performance.png` - Monthly net P&L breakdown
- `cost_breakdown.png` - Spread/slippage/commission analysis
- `funnel_analysis.png` - Pattern filtering funnel
- `report.txt` - Detailed text statistics including:
  - Gross vs Net P&L
  - Execution cost breakdown
  - Exit reason analysis
  - Filter rejection stats
  - Confluence scoring
- `results.json` - Raw result payload for reproducible analysis

Notes:
- Using `--no-charts` still writes `report.txt`, `amd_conformity.json`, and `results.json`.
- Use `--no-report` to skip all report artifact output.

## Judas Swing Quality Filter

The manipulation detection includes Judas swing quality scoring to filter out low-quality fakeouts:

| Factor | Scoring | Config Key |
|--------|---------|------------|
| Candle count | 1-5 candles = fast sweep = +1 | `judas_max_candles` |
| Velocity | Break distance per candle >= 0.3 ATR = +1 | `judas_min_velocity_atr` |
| Session | London open (07:00-10:00 UTC) = +1 | `judas_london_bonus` |
| Midnight sweep | Price swept above/below midnight open | tracked but not scored |

Short trades require a minimum confluence score of 2 (configurable via `short_min_confluence_score`), which naturally filters out low-quality short setups that historically showed a 36.9% win rate vs 51% for longs.

## Monte Carlo Simulation

The Monte Carlo analysis reshuffles the actual trade sequence 10,000 times to estimate:

- **Drawdown distribution**: P5/P25/P50/P75/P95 of max drawdown
- **Ruin probability**: Chance of hitting 20%, 30%, 50%, 80% drawdown thresholds
- **Time to double**: Average trades needed to double the account
- **Consecutive loss streaks**: Probability of N+ consecutive losses

Target: P95 max drawdown should be ~25% (even in the worst 5% of shuffles, the account survives).

Reports include fan charts (equity envelopes), drawdown histograms, and ruin probability bar charts.

## Live Signal Scanner

The live module scans real-time candle data for AMD patterns using the same strategy pipeline as the backtester.

**Signal-only mode** (default): No MT5 execution -- signals are printed to console and optionally logged to a JSONL file.

Kill switch rules:
- Daily loss > 1% of initial balance -> stop for the day
- Account drawdown > 15% -> halt until manual review
- Consecutive choppy regime checks >= 2 -> pause scanning

Each signal includes: direction, entry price, SL, TP, position size, R:R, confluence score, Judas quality, and confidence level (HIGH/MEDIUM/LOW).

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_strategy.py::TestRisk -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Example Output

```
============================================================
BACKTEST RESULTS
============================================================
Backtest ID:      abc12345
Total Trades:     847
Win Rate:         48.2%
Expectancy:       0.312R
Profit Factor:    1.45
Max Drawdown:     8.7%
============================================================

P&L BREAKDOWN:
  Gross P&L:      $12,450.00
  Net P&L:        $9,823.00
  Final Capital:  $19,823.00

EXECUTION COSTS:
  Spread:         $1,271.00
  Slippage:       $892.00
  Commission:     $464.00
  Total Costs:    $2,627.00
============================================================
```

## License

MIT License - Use at your own risk. Past performance does not guarantee future results. This software is for educational and research purposes only.
