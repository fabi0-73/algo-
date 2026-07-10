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

from config import STRATEGY, RISK_MODEL, SESSION_FILTER, CONFIDENCE_SIZING, ADAPTIVE_EXITS, SIGNAL_CONFIDENCE
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
    calculate_move_potential,
    calculate_signal_confidence,
    EntrySignal,
    ENTRY_MODE_RETEST_ONLY,
)
from src.strategy.consolidation import score_consolidation_quality
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

    # Empirical signal confidence (0-4 + LOW/MODERATE/GOOD/HIGH label)
    signal_confidence: int = 0
    confidence_label: str = ""

    # Adaptive exit guidance
    move_potential: int = 0          # 0-5: how far the setup may run
    exit_tier: str = ""              # "runner", "standard", or "" (default)
    suggested_tp: float = 0.0       # Tier-adjusted TP (0 = trail only)
    trailing_activation_r: float = 0.0
    trailing_atr_mult: float = 0.0
    be_trigger_r: float = 0.0
    partial_tp_at_r: float = 0.0
    partial_close_pct: float = 0.0

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

        # NY_IB stream: day-scoped dedup (one attempt per broker day — the
        # 30-min price-keyed AMD dedup is unsuitable for a daily setup)
        self._nyib_signaled_days: set = set()

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

        for offset in range(min_offset, max_offset,
                            STRATEGY.get("consolidation_scan_step", 2)):
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
            if not validate_distribution_strength(
                    df, dist,
                    min_follow_through_candles=STRATEGY.get("distribution_follow_through_candles", 2),
                    current_idx=current_idx):
                continue

            bos = None
            entry_mode = STRATEGY.get("entry_mode", ENTRY_MODE_RETEST_ONLY)
            # Engine parity (engine.py _scan_for_patterns): BOS must be searched
            # whenever bos_required is set, regardless of entry mode. The old
            # `entry_mode != RETEST_ONLY` condition meant that under the shipping
            # config (RETEST_ONLY + bos_required) live NEVER found a BOS, and the
            # entry gate then rejected every AMD setup — zero live signals.
            if STRATEGY.get("bos_required", False) or entry_mode != ENTRY_MODE_RETEST_ONLY:
                expected_dir = "BULLISH" if manip.direction == "DOWN" else "BEARISH"
                bos = find_bos_after_manipulation(
                    df, manip.return_candle_idx, expected_dir,
                    search_window=20,
                    swing_lookback=STRATEGY.get("bos_swing_lookback", 5),
                    current_idx=current_idx,
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

            # Compute move potential and resolve exit tier
            entry_hour = ts.hour if hasattr(ts, "hour") else 0
            equal_level_swept = getattr(entry, "equal_level_swept", False)
            mp = calculate_move_potential(
                velocity_score=getattr(manip, "velocity_score", 0.0),
                session_hour=entry_hour,
                body_expansion=getattr(dist, "body_expansion", 0.0),
                consolidation_quality=score_consolidation_quality(consol),
                equal_level_swept=equal_level_swept,
            )
            exit_tier_name = ""
            tier_cfg = {}
            for tier in ADAPTIVE_EXITS.get("tiers", []):
                if mp >= tier.get("min_move_potential", 999):
                    exit_tier_name = tier["name"]
                    tier_cfg = tier
                    break

            # Empirical signal confidence (same function as backtest) + HIGH up-sizing
            signal_conf, conf_label = calculate_signal_confidence(confluence, mp, entry_hour)
            if (SIGNAL_CONFIDENCE.get("enabled", False)
                    and SIGNAL_CONFIDENCE.get("size_by_confidence", False)):
                extra = 0.0
                if conf_label == "HIGH":
                    extra = SIGNAL_CONFIDENCE.get("high_extra_lots", 0.0)
                elif conf_label == "GOOD":
                    extra = SIGNAL_CONFIDENCE.get("good_extra_lots", 0.0)
                if extra > 0:
                    risk.position_size = min(
                        risk.position_size + extra, RISK_MODEL.get("max_lot", 50.0)
                    )

            # Compute tier-adjusted TP
            stop_dist = abs(risk.entry_price - risk.stop_loss) if hasattr(risk, "entry_price") else abs(entry.entry_price - risk.stop_loss)
            suggested_tp = risk.take_profit
            if tier_cfg.get("tp_rr", 0) > 0 and stop_dist > 0:
                if entry.direction == "LONG":
                    suggested_tp = entry.entry_price + stop_dist * tier_cfg["tp_rr"]
                else:
                    suggested_tp = entry.entry_price - stop_dist * tier_cfg["tp_rr"]
            elif tier_cfg.get("tp_rr", -1) == 0:
                suggested_tp = 0.0  # No static TP — trail only

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
                signal_confidence=signal_conf,
                confidence_label=conf_label,
                # Adaptive exit guidance
                move_potential=mp,
                exit_tier=exit_tier_name,
                suggested_tp=round(suggested_tp, 2),
                trailing_activation_r=tier_cfg.get("trailing_activation_r", 0),
                trailing_atr_mult=tier_cfg.get("trailing_atr_mult", 0),
                be_trigger_r=tier_cfg.get("be_trigger_r", 0),
                partial_tp_at_r=tier_cfg.get("partial_tp_at_r", 0),
                partial_close_pct=tier_cfg.get("partial_close_pct", 0),
                # Pattern context
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
            break  # One AMD signal per scan

        # NY_IB stream runs beside AMD — both may emit in the same scan
        try:
            signals.extend(self.scan_ny_ib(df))
        except Exception as e:  # a broken second stream must never kill AMD
            logger.error(f"NY_IB scan failed: {e}")

        return signals

    def scan_ny_ib(self, df: pd.DataFrame) -> List[LiveSignal]:
        """NY Initial Balance pullback signals (NY_IB_MODEL; engine-validated
        runs 19af4776/12a01408: 77% WR, PF 1.8, +0.10R after news filter).

        IB = 16:30-17:30 broker high/low from CLOSED bars (run_live drops the
        forming bar). After a bar closes beyond the IB within 17:30-22:00, emit
        ONE signal per day: a LIMIT back inside the range, SL across the range,
        small TP past the edge, force-flat 23:00. The breakout close must be on
        one of the last few bars — stale breakouts (scanner offline) are skipped.
        """
        from config import NY_IB_MODEL

        if not NY_IB_MODEL.get("enabled", False):
            return []

        def _mins(s):
            hh, mm = (int(x) for x in str(s).split(":"))
            return hh * 60 + mm

        ib_start = _mins(NY_IB_MODEL.get("ib_start", "16:30"))
        ib_end = _mins(NY_IB_MODEL.get("ib_end", "17:30"))
        scan_end = _mins(NY_IB_MODEL.get("scan_end", "22:00"))

        ts_last = df["timestamp"].iloc[-1]
        day = pd.Timestamp(ts_last).normalize()
        if day in self._nyib_signaled_days:
            return []

        mod = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
        today = df["timestamp"].dt.normalize() == day
        ib_bars = df[today & (mod >= ib_start) & (mod < ib_end)]
        if len(ib_bars) < int(NY_IB_MODEL.get("min_ib_bars", 10)):
            return []
        ib_hi = float(ib_bars["high"].max())
        ib_lo = float(ib_bars["low"].min())
        size = ib_hi - ib_lo
        ref = float(ib_bars["close"].iloc[-1])
        if not (NY_IB_MODEL.get("ib_min_pct", 0.004) * ref
                <= size <= NY_IB_MODEL.get("ib_max_pct", 0.02) * ref):
            return []

        window = df[today & (mod >= ib_end) & (mod < scan_end)]
        if window.empty:
            return []
        long_only = NY_IB_MODEL.get("long_only", False)
        breaks = window[(window["close"] > ib_hi)
                        | ((window["close"] < ib_lo) if not long_only else False)]
        if breaks.empty:
            return []
        first = breaks.iloc[0]
        # The FIRST confirming close consumes the day's attempt (engine/lab
        # parity) — mark the day even if the signal below is rejected.
        self._nyib_signaled_days.add(day)
        if len(self._nyib_signaled_days) > 10:
            self._nyib_signaled_days = set(sorted(self._nyib_signaled_days)[-10:])

        # Act only on FRESH breakouts (within the last 3 closed bars) — a stale
        # limit price from hours ago is not a tradeable signal.
        if first.name < len(df) - 3:
            logger.info("NY_IB: breakout close is stale (scanner gap?) — skipped")
            return []

        # News blackout still applies; killzone gate deliberately does not
        can_enter, reason = self.news_filter.can_enter_trade(first["timestamp"])
        if not can_enter:
            logger.info(f"NY_IB: breakout filtered by news blackout ({reason})")
            return []

        direction = "LONG" if float(first["close"]) > ib_hi else "SHORT"
        rf = NY_IB_MODEL.get("retrace_frac", 0.10)
        slm = NY_IB_MODEL.get("sl_range_mult", 1.0)
        tpm = NY_IB_MODEL.get("tp_range_mult", 0.20)
        if direction == "LONG":
            limit = ib_hi - rf * size
            sl = limit - slm * size
            tp = ib_hi + tpm * size
        else:
            limit = ib_lo + rf * size
            sl = limit + slm * size
            tp = ib_lo - tpm * size

        # Flat base-tier sizing (no AMD confluence), engine parity
        base_risk = CONFIDENCE_SIZING.get("base_risk_pct",
                                          RISK_MODEL.get("risk_pct_per_trade_default", 0.005))
        stop_dist = abs(limit - sl)
        contract = RISK_MODEL.get("contract_size", 100)
        step = RISK_MODEL.get("lot_step", 0.01)
        lots = (self.account_balance * base_risk) / (stop_dist * contract)
        lots = round(lots / step) * step
        lots = min(max(lots, RISK_MODEL.get("min_lot", 0.01)),
                   RISK_MODEL.get("max_lot", 1.0))

        signal = LiveSignal(
            timestamp=first["timestamp"],
            symbol=self.symbol,
            direction=direction,
            entry_price=round(limit, 2),
            stop_loss=round(sl, 2),
            take_profit=round(tp, 2),
            position_size_lots=lots,
            risk_reward=round(abs(tp - limit) / stop_dist, 2) if stop_dist > 0 else 0.0,
            confluence_score=0,
            judas_quality=0,
            confidence="base",
            risk_pct=base_risk,
            entry_mode="NY_IB",
            consolidation_high=ib_hi,
            consolidation_low=ib_lo,
        )
        logger.info(
            f"NY_IB SIGNAL: {direction} limit {limit:.2f} "
            f"(IB {ib_lo:.2f}-{ib_hi:.2f}, SL {sl:.2f}, TP {tp:.2f}, {lots:.2f} lots)"
        )
        return [signal]

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
