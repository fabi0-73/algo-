"""
Live Signal Scanner

Real-time AMD pattern scanner that reuses the backtest strategy modules.
Scans the latest N candles for complete AMD patterns and returns signal objects.
"""
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

from config import STRATEGY, RISK_MODEL, SESSION_FILTER, CONFIDENCE_SIZING
from src.strategy.indicators import add_indicators, calculate_atr
from src.strategy.consolidation import ConsolidationResult, detect_equal_levels
from src.strategy.manipulation import (
    ManipulationResult,
    confirm_liquidity_sweep,
    confirm_volume_spike,
    score_judas_quality,
)
from src.strategy.distribution import DistributionResult, validate_distribution_strength
from src.strategy.entry import (
    check_entry_at_candle,
    check_premium_discount_filter,
    EntrySignal,
    ENTRY_MODE_RETEST_ONLY,
)
from src.strategy.risk import calculate_risk, RiskParams
from src.strategy.fvg import find_fvg_at_retest_level
from src.strategy.order_blocks import find_ob_at_retest_level
from src.strategy.market_structure import find_bos_after_manipulation
from src.strategy.time_filters import TimeFilterEngine, localize_dataframe_timestamps
from src.strategy.news_filter import NewsFilterEngine
from src.strategy.htf_bias import HTFBiasEngine
from src.strategy.volume_filters import VolumeFilterEngine

logger = logging.getLogger(__name__)


@dataclass
class LiveSignal:
    """A live trading signal with all parameters needed for execution."""
    timestamp: datetime
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_lots: float
    risk_reward: float
    confluence_score: int
    judas_quality: int
    confidence: str

    risk_pct: float = 0.0

    consolidation_high: float = 0.0
    consolidation_low: float = 0.0
    manipulation_extreme: float = 0.0
    entry_mode: str = ""
    fvg_confluence: bool = False
    ob_confluence: bool = False
    bos_confirmed: bool = False
    volume_confirmed: bool = False
    midnight_price_swept: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        return d


class LiveSignalScanner:
    """
    Scans live candle data for AMD pattern signals.

    Reuses the same consolidation -> manipulation -> distribution -> entry
    pipeline as the backtest engine, but operates on the most recent window.
    """

    def __init__(
        self,
        symbol: str = None,
        account_balance: float = None,
    ):
        self.symbol = symbol or STRATEGY["symbol"]
        self.account_balance = account_balance or 100.0

        self.lookback = STRATEGY["consolidation_lookback"]
        self.atr_period = STRATEGY["atr_period"]
        self.min_bars = self.lookback + self.atr_period + 60

        self.time_filter = TimeFilterEngine()
        self.news_filter = NewsFilterEngine()
        self.htf_bias = HTFBiasEngine()
        self.volume_filter = VolumeFilterEngine()

        # Deduplication: track recently emitted signals by pattern key
        self._recent_signals: dict[str, datetime] = {}
        self._signal_expiry_minutes = 30  # Ignore duplicate within this window

    def scan(self, df: pd.DataFrame) -> List[LiveSignal]:
        """
        Scan a DataFrame of recent candles for AMD signals.

        Args:
            df: DataFrame with OHLC + tick_volume + timestamp columns.
                Must have at least self.min_bars rows.

        Returns:
            List of LiveSignal objects (may be empty).
        """
        if len(df) < self.min_bars:
            logger.warning(f"Need at least {self.min_bars} candles, got {len(df)}")
            return []

        df = add_indicators(df.copy(), atr_period=self.atr_period)
        df = df.reset_index(drop=True)
        df = localize_dataframe_timestamps(df)
        df = self.htf_bias.add_htf_bias(df)
        df = self.news_filter.add_blackout_column(df)

        current_idx = len(df) - 1
        signals: List[LiveSignal] = []

        atr = df["atr"].iloc[current_idx]
        if pd.isna(atr) or atr == 0:
            return []

        row = df.iloc[current_idx]
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        min_offset = STRATEGY.get("pattern_min_bars_after_consolidation", 10)
        max_offset = min(
            STRATEGY.get("pattern_max_bars_after_consolidation", 60),
            current_idx - self.lookback,
        )

        for offset in range(min_offset, max_offset, 2):
            end_idx = current_idx - offset
            start_idx = end_idx - self.lookback

            if start_idx < self.atr_period:
                continue

            hw = highs[start_idx:end_idx + 1]
            lw = lows[start_idx:end_idx + 1]
            cw = closes[start_idx:end_idx + 1]

            if not self._is_consolidation(hw, lw, cw, atr):
                continue

            range_high = float(np.max(hw))
            range_low = float(np.min(lw))
            consol = ConsolidationResult(
                valid=True,
                range_high=range_high,
                range_low=range_low,
                range_size=range_high - range_low,
                atr=atr,
                start_idx=start_idx,
                end_idx=end_idx,
            )
            if STRATEGY.get("detect_equal_levels", True):
                consol = detect_equal_levels(df, consol)

            manip = self._find_manipulation(df, consol, current_idx)
            if not manip.valid:
                continue

            manip = score_judas_quality(df, manip)
            manip = confirm_volume_spike(df, manip)

            dist = self._find_distribution(df, consol, manip, current_idx)
            if not dist.valid:
                continue
            if not validate_distribution_strength(df, dist, min_follow_through_candles=2):
                continue

            bos = None
            entry_mode = STRATEGY.get("entry_mode", ENTRY_MODE_RETEST_ONLY)
            if entry_mode != ENTRY_MODE_RETEST_ONLY:
                expected_dir = "BULLISH" if manip.direction == "DOWN" else "BEARISH"
                bos = find_bos_after_manipulation(
                    df, manip.return_candle_idx, expected_dir,
                    search_window=20,
                    swing_lookback=STRATEGY.get("bos_swing_lookback", 5),
                )

            entry = check_entry_at_candle(
                df=df, current_idx=current_idx,
                consolidation=consol, manipulation=manip,
                distribution=dist, structure_break=bos,
                entry_mode=entry_mode,
            )
            if not entry.valid:
                continue

            # Apply filters
            ts = row.get("timestamp")
            can_enter, _ = self.time_filter.can_enter_trade(ts, self.account_balance)
            if not can_enter:
                continue
            can_enter, _ = self.news_filter.can_enter_trade(ts)
            if not can_enter:
                continue
            can_enter, _, _ = self.htf_bias.can_enter_trade(entry.direction, df, current_idx)
            if not can_enter:
                continue

            if consol.start_idx is not None and dist.break_candle_idx is not None:
                can_enter, _, _ = self.volume_filter.can_enter_trade(
                    df, dist.break_candle_idx, consol.start_idx, consol.end_idx,
                )
                if not can_enter:
                    continue

            risk = calculate_risk(entry, self.account_balance, atr=atr)
            if not risk.valid:
                continue

            confluence = getattr(entry, "confluence_score", 0)
            judas_q = getattr(manip, "judas_quality", 0)

            # Match backtest CONFIDENCE_SIZING tiers
            confidence, risk_pct = self._classify_confidence_tier(
                confluence, ts,
            )

            signal = LiveSignal(
                timestamp=ts,
                symbol=self.symbol,
                direction=entry.direction,
                entry_price=round(entry.desired_entry_price or entry.entry_price, 2),
                stop_loss=risk.stop_loss,
                take_profit=risk.take_profit,
                position_size_lots=risk.position_size,
                risk_reward=risk.risk_reward_ratio,
                confluence_score=confluence,
                judas_quality=judas_q,
                confidence=confidence,
                risk_pct=risk_pct,
                consolidation_high=consol.range_high,
                consolidation_low=consol.range_low,
                manipulation_extreme=manip.extreme_price,
                entry_mode=getattr(entry, "entry_mode", ""),
                fvg_confluence=getattr(entry, "fvg_confluence", False),
                ob_confluence=getattr(entry, "ob_confluence", False),
                bos_confirmed=getattr(entry, "bos_confirmed", False),
                volume_confirmed=getattr(manip, "volume_confirmed", False),
                midnight_price_swept=getattr(manip, "midnight_price_swept", False),
            )
            # Deduplication: skip if same pattern seen recently
            sig_key = f"{signal.direction}_{signal.entry_price:.2f}_{consol.range_high:.2f}_{consol.range_low:.2f}"
            now = datetime.now()
            if sig_key in self._recent_signals:
                age = (now - self._recent_signals[sig_key]).total_seconds() / 60
                if age < self._signal_expiry_minutes:
                    logger.debug(f"Skipping duplicate signal {sig_key} (age {age:.0f}m)")
                    break

            self._recent_signals[sig_key] = now
            # Prune expired entries
            self._recent_signals = {
                k: v for k, v in self._recent_signals.items()
                if (now - v).total_seconds() / 60 < self._signal_expiry_minutes
            }

            signals.append(signal)
            break  # One signal per scan

        return signals

    @staticmethod
    def _classify_confidence_tier(confluence_score: int, ts) -> tuple:
        """Classify trade into confidence tier using CONFIDENCE_SIZING config.

        Returns (tier_name, risk_pct).
        """
        if not CONFIDENCE_SIZING.get("enabled", False):
            base = RISK_MODEL.get("risk_pct_per_trade_default", 0.005)
            return "base", base

        # Determine if current hour is within prime hours
        prime_start = CONFIDENCE_SIZING.get("prime_hours_start", 13)
        prime_end = CONFIDENCE_SIZING.get("prime_hours_end", 17)
        hour = ts.hour if hasattr(ts, "hour") else 0
        is_prime = prime_start <= hour <= prime_end

        # Check tiers top-down, first match wins
        for tier in CONFIDENCE_SIZING.get("tiers", []):
            min_score = tier.get("min_confluence_score", 0)
            prime_only = tier.get("prime_hours_only", False)

            if confluence_score >= min_score:
                if prime_only and not is_prime:
                    continue
                return tier["name"], tier["risk_pct"]

        return "base", CONFIDENCE_SIZING.get("base_risk_pct", 0.003)

    def _is_consolidation(
        self, h: np.ndarray, l: np.ndarray, c: np.ndarray, atr: float,
    ) -> bool:
        rng = float(np.max(h)) - float(np.min(l))
        if rng > STRATEGY["consolidation_range_atr_mult"] * atr:
            return False
        inside = np.sum((c >= np.min(l)) & (c <= np.max(h)))
        return (inside / len(c)) >= STRATEGY["consolidation_close_pct"]

    def _find_manipulation(
        self, df: pd.DataFrame, consol: ConsolidationResult, current_idx: int,
    ) -> ManipulationResult:
        h = df["high"].values
        l_ = df["low"].values
        c = df["close"].values
        atr = consol.atr
        min_break = STRATEGY["manipulation_break_atr_mult"] * atr
        max_ret = STRATEGY["manipulation_return_candles"]

        search_start = consol.end_idx + 1
        search_end = min(current_idx, search_start + 30)

        for direction in ("UP", "DOWN"):
            break_idx = -1
            extreme = 0.0 if direction == "UP" else float("inf")
            for i in range(search_start, search_end):
                if direction == "UP":
                    if h[i] > consol.range_high + min_break:
                        if break_idx == -1:
                            break_idx = i
                        extreme = max(extreme, h[i])
                    if break_idx >= 0:
                        if c[i] <= consol.range_high and c[i] >= consol.range_low:
                            if i - break_idx <= max_ret:
                                return ManipulationResult(
                                    valid=True, direction="UP",
                                    extreme_price=extreme,
                                    break_distance=extreme - consol.range_high,
                                    return_candle_idx=i, atr=atr,
                                    manipulation_candle_count=max(1, i - break_idx),
                                )
                        if i - break_idx > max_ret:
                            break_idx = -1
                            extreme = 0.0
                else:
                    if l_[i] < consol.range_low - min_break:
                        if break_idx == -1:
                            break_idx = i
                        extreme = min(extreme, l_[i])
                    if break_idx >= 0:
                        if c[i] >= consol.range_low and c[i] <= consol.range_high:
                            if i - break_idx <= max_ret:
                                return ManipulationResult(
                                    valid=True, direction="DOWN",
                                    extreme_price=extreme,
                                    break_distance=consol.range_low - extreme,
                                    return_candle_idx=i, atr=atr,
                                    manipulation_candle_count=max(1, i - break_idx),
                                )
                        if i - break_idx > max_ret:
                            break_idx = -1
                            extreme = float("inf")

        return ManipulationResult(valid=False)

    def _find_distribution(
        self, df: pd.DataFrame, consol: ConsolidationResult,
        manip: ManipulationResult, current_idx: int,
    ) -> DistributionResult:
        o = df["open"].values
        c = df["close"].values
        atr = manip.atr
        min_break = STRATEGY["distribution_break_atr_mult"] * atr
        body_mult = STRATEGY["distribution_body_mult"]

        s, e = consol.start_idx, consol.end_idx + 1
        avg_body = float(np.mean(np.abs(c[s:e] - o[s:e]))) or atr * 0.1

        expected_dir = "UP" if manip.direction == "DOWN" else "DOWN"
        search_start = manip.return_candle_idx + 1
        search_end = min(current_idx, search_start + 20)

        for i in range(search_start, search_end):
            body = abs(c[i] - o[i])
            ratio = body / avg_body if avg_body > 0 else 0

            if expected_dir == "UP":
                bd = c[i] - consol.range_high
                if bd >= min_break and ratio >= body_mult:
                    return DistributionResult(
                        valid=True, direction="UP", break_price=c[i],
                        break_distance=bd, body_expansion=ratio,
                        break_candle_idx=i, atr=atr,
                    )
            else:
                bd = consol.range_low - c[i]
                if bd >= min_break and ratio >= body_mult:
                    return DistributionResult(
                        valid=True, direction="DOWN", break_price=c[i],
                        break_distance=bd, body_expansion=ratio,
                        break_candle_idx=i, atr=atr,
                    )

        return DistributionResult(valid=False)
