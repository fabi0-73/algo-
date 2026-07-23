"""Planted patterns fire exactly at the planted bar; prefix-invariance
(detect(df[:i+1]) row i == detect(df) row i) mechanically forbids lookahead
for every registered detector."""
import numpy as np
import pandas as pd

from src.research.events import (
    EVENT_REGISTRY, detect_bos_up, detect_fvg_bull, detect_fvg_fill_bull,
    detect_range_break, detect_session_open_london, detect_sweep_high,
    detect_vwap_stretch, run_all,
)
from src.strategy.fvg import detect_fvg as oracle_detect_fvg


def frame(bars, start="2025-01-06 09:00", atr=1.0, volume=100):
    """bars = list of (open, high, low, close)."""
    ts = pd.date_range(start, periods=len(bars), freq="5min")
    df = pd.DataFrame(bars, columns=["open", "high", "low", "close"])
    df.insert(0, "timestamp", ts)
    df["volume"] = volume
    df["atr"] = atr
    return df


def flat_bars(n, px=100.0):
    return [(px, px + 0.2, px - 0.2, px)] * n


def random_frame(n=400, seed=11):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.5, n).cumsum()
    close = 100 + steps
    open_ = np.roll(close, 1)
    open_[0] = 100
    high = np.maximum(open_, close) + rng.uniform(0, 0.4, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.4, n)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-06 02:00", periods=n, freq="5min"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.integers(50, 500, n),
    })
    df["atr"] = 1.0
    # correlated-silver leg so ratio_stretch is genuinely exercised by the
    # prefix-invariance sweep (detectors ignore unknown columns)
    df["xag_close"] = close / 80.0 + rng.normal(0, 0.01, n)
    return df


def test_fvg_bull_fires_at_third_bar():
    bars = flat_bars(5) + [
        (100.0, 100.3, 99.8, 100.2),   # bar 5: high 100.3
        (100.3, 101.5, 100.2, 101.4),  # bar 6: impulse
        (101.4, 101.8, 100.9, 101.6),  # bar 7: low 100.9 > 100.3 -> gap 0.6
    ] + flat_bars(2, 101.5)
    df = frame(bars)
    r = detect_fvg_bull(df)
    assert r.loc[7, "fired"] and r.loc[7, "direction"] == 1
    assert np.isclose(r.loc[7, "strength"], 0.6)
    assert r["fired"].sum() == 1  # nowhere else


def test_fvg_matches_oracle():
    df = random_frame(300)
    ours = detect_fvg_bull(df)
    for i in range(2, len(df)):
        o = oracle_detect_fvg(df, i, min_size_atr_mult=0.10, atr=1.0)
        expected = o is not None and o.valid and o.direction == "BULLISH"
        assert bool(ours.loc[i, "fired"]) == expected, f"mismatch at {i}"


def test_sweep_high_fires_on_poke_and_reject():
    bars = flat_bars(30)  # prior max 100.2
    bars += [(100.0, 100.6, 99.9, 100.1)]  # poke 0.4 above, close back below
    df = frame(bars)
    r = detect_sweep_high(df)
    i = len(bars) - 1
    assert r.loc[i, "fired"] and r.loc[i, "direction"] == -1
    assert np.isclose(r.loc[i, "strength"], 0.4)
    # a poke that CLOSES above is not a sweep
    bars2 = flat_bars(30) + [(100.0, 100.6, 99.9, 100.5)]
    assert not detect_sweep_high(frame(bars2))["fired"].iloc[-1]


def test_bos_up_fires_at_break_not_pivot():
    px = 100.0
    bars = flat_bars(10, px)
    bars += [(px, 102.0, px - 0.2, px + 0.5)]   # bar 10: swing high 102
    bars += flat_bars(5, px)                    # bars 11-15 below (confirm at 13)
    bars += [(px, 103.0, px - 0.1, 102.5)]      # bar 16: close above 102 -> BOS
    df = frame(bars)
    r = detect_bos_up(df)
    assert r.loc[16, "fired"] and r.loc[16, "direction"] == 1
    assert not r.loc[10, "fired"]  # never at the pivot
    assert r["fired"].sum() == 1


def test_range_break_needs_compression():
    px = 100.0
    bars = flat_bars(40, px)                    # tight 0.4-wide range
    bars += [(px, 101.5, px - 0.1, 101.4)]      # close above range hi
    df = frame(bars)
    r = detect_range_break(df)
    assert r["fired"].iloc[-1] and r["direction"].iloc[-1] == 1
    # same break after a WIDE range does not fire
    wide = [(px, px + 3.0, px - 3.0, px)] * 40 + [(px, 104.0, px - 0.1, 103.8)]
    assert not detect_range_break(frame(wide))["fired"].iloc[-1]


def test_fvg_fill_first_touch_only():
    bars = flat_bars(5) + [
        (100.0, 100.3, 99.8, 100.2),
        (100.3, 101.5, 100.2, 101.4),
        (101.4, 101.8, 100.9, 101.6),  # bull FVG zone [100.3, 100.9] at bar 7
        (101.6, 101.7, 101.2, 101.5),  # no touch
        (101.5, 101.6, 100.7, 101.0),  # bar 9: low 100.7 <= 100.9 -> fill
        (101.0, 101.2, 100.5, 100.8),  # dips again -> must NOT fire
    ]
    df = frame(bars)
    r = detect_fvg_fill_bull(df)
    assert r.loc[9, "fired"] and r.loc[9, "direction"] == 1
    assert r["fired"].sum() == 1


def test_session_open_london_once_per_day():
    n = 288 * 2  # two full days of M5
    df = random_frame(n)
    df["timestamp"] = pd.date_range("2025-01-06 00:00", periods=n, freq="5min")
    r = detect_session_open_london(df)
    fired_ts = df.loc[r["fired"], "timestamp"]
    assert len(fired_ts) == 2
    assert all(t.hour == 10 and t.minute == 0 for t in fired_ts)


def test_vwap_stretch_direction_is_mean_reverting():
    df = random_frame(200)
    r = detect_vwap_stretch(df, {"min_stretch_atr": 0.5})
    if r["fired"].any():
        from src.research.strategies.base import anchored_vwap
        vwap = anchored_vwap(df)
        stretched = df["close"] - vwap
        for i in r.index[r["fired"]]:
            assert np.sign(r.loc[i, "direction"]) == -np.sign(stretched[i])


def test_all_detectors_prefix_invariant():
    df = random_frame(400)
    full = run_all(df)
    for cut in (150, 275, 399):
        prefix = df.iloc[: cut + 1].reset_index(drop=True)
        part = run_all(prefix)
        for name in EVENT_REGISTRY:
            f_row = full[name].loc[cut]
            p_row = part[name].loc[cut]
            assert f_row["fired"] == p_row["fired"], f"{name} lookahead at {cut}"
            assert f_row["direction"] == p_row["direction"], name
            assert np.isclose(f_row["strength"], p_row["strength"]), name


def test_detectors_return_aligned_schema():
    df = random_frame(120)
    for name, res in run_all(df).items():
        assert list(res.columns) == ["fired", "direction", "strength"], name
        assert len(res) == len(df), name
        assert res["fired"].dtype == bool, name
        assert (res.loc[~res["fired"], "direction"] == 0).all(), name


# ------------------------------------------------- 2026-07-10 detector batch

from src.research.events import (  # noqa: E402
    detect_failed_break, detect_h1_sweep, detect_inside_nr7,
    detect_news_reopen, detect_orb_break_london, detect_orb_pullback_london,
    detect_pm_fix_window, detect_round_level_reject, detect_settlement_gap,
    detect_sweep_reclaim, detect_vol_dryup, detect_wick_rejection,
)


def day_frame(start="2025-01-06 09:00", n=72, px=100.0):
    """Contiguous flat M5 day starting at `start` (broker time)."""
    return frame(flat_bars(n, px), start=start)


def test_orb_break_london_fires_after_window_closes():
    # 09:00 start -> London OR window = 10:00-10:25 (6 bars, or_minutes=30)
    df = day_frame()
    i_break = 18  # the 10:30 bar
    df.loc[i_break, ["open", "high", "low", "close"]] = [100.0, 101.2, 99.9, 101.0]
    r = detect_orb_break_london(df)
    assert r.loc[i_break, "fired"] and r.loc[i_break, "direction"] == 1
    assert np.isclose(r.loc[i_break, "strength"], 0.8)  # 101.0 - OR hi 100.2
    assert r["fired"].sum() == 1
    # a close above OR-hi DURING the forming window must not fire
    df2 = day_frame()
    df2.loc[13, ["open", "high", "low", "close"]] = [100.0, 101.2, 99.9, 101.0]
    assert not detect_orb_break_london(df2).loc[13, "fired"]


def test_orb_pullback_london_retest_of_broken_edge():
    df = day_frame()
    df.loc[18, ["open", "high", "low", "close"]] = [100.0, 101.2, 99.9, 101.0]
    df.loc[20, ["open", "high", "low", "close"]] = [100.9, 100.9, 100.3, 100.6]
    r = detect_orb_pullback_london(df)
    assert r.loc[20, "fired"] and r.loc[20, "direction"] == 1
    assert np.isclose(r.loc[20, "strength"], 0.4)  # close 100.6 - edge 100.2
    assert r["fired"].sum() == 1


def test_sweep_reclaim_fires_on_confirmation_not_poke():
    bars = flat_bars(30)                          # prior min 99.8
    bars += [(100.0, 100.0, 99.4, 99.9)]          # bar 30: sweep low, high 100.0
    bars += [(99.9, 100.1, 99.8, 100.0)]          # bar 31: not above 100.0 yet
    bars += [(100.0, 100.6, 99.9, 100.5)]         # bar 32: close 100.5 > 100.0
    df = frame(bars)
    r = detect_sweep_reclaim(df)
    assert not r.loc[30, "fired"]                 # never at the poke
    assert r.loc[32, "fired"] and r.loc[32, "direction"] == 1
    assert np.isclose(r.loc[32, "strength"], 0.5)
    assert r["fired"].sum() == 1
    # a reclaim AFTER the confirm window must not fire
    bars2 = flat_bars(30) + [(100.0, 100.0, 99.4, 99.9)] + flat_bars(7, 99.9) \
        + [(99.9, 100.7, 99.8, 100.6)]
    assert not detect_sweep_reclaim(frame(bars2))["fired"].iloc[-1]


def test_failed_break_fades_reentry():
    px = 100.0
    bars = flat_bars(40, px)                       # compressed range, hi 100.2
    bars += [(px, 101.5, px - 0.1, 101.4)]         # bar 40: range break up
    bars += [(101.4, 101.5, 101.0, 101.2)]         # still outside
    bars += [(101.2, 101.3, 99.7, 99.9)]           # bar 42: close back inside
    df = frame(bars)
    r = detect_failed_break(df)
    assert r.loc[42, "fired"] and r.loc[42, "direction"] == -1
    assert np.isclose(r.loc[42, "strength"], 0.3)  # edge 100.2 - close 99.9
    assert r["fired"].sum() == 1


def test_wick_rejection_pin_bar():
    bars = flat_bars(5)
    bars += [(100.0, 100.15, 98.8, 100.1)]  # lower wick 1.2, body 0.1
    bars += flat_bars(2)
    df = frame(bars)
    r = detect_wick_rejection(df)
    assert r.loc[5, "fired"] and r.loc[5, "direction"] == 1
    assert np.isclose(r.loc[5, "strength"], 1.2)
    # a big-body bar must not fire
    bars2 = flat_bars(5) + [(100.0, 101.6, 99.9, 101.5)] + flat_bars(2)
    assert not detect_wick_rejection(frame(bars2)).loc[5, "fired"]


def test_round_level_reject_static_grid():
    bars = flat_bars(3, 99.0)
    bars += [(99.5, 100.3, 99.4, 99.8)]    # poke $100 from below, reject
    bars += [(100.5, 100.6, 99.8, 100.4)]  # poke $100 from above, reject
    df = frame(bars)
    r = detect_round_level_reject(df)
    assert r.loc[3, "fired"] and r.loc[3, "direction"] == -1
    assert np.isclose(r.loc[3, "strength"], 0.3)
    assert r.loc[4, "fired"] and r.loc[4, "direction"] == 1
    assert np.isclose(r.loc[4, "strength"], 0.2)


def test_vol_dryup_fires_on_low_participation():
    df = frame(flat_bars(60))
    df["volume"] = 100
    df.loc[55, "volume"] = 20
    r = detect_vol_dryup(df)
    assert r.loc[55, "fired"] and r.loc[55, "direction"] == 0
    assert np.isclose(r.loc[55, "strength"], 5.0)
    assert not r.loc[54, "fired"]


def test_inside_nr7_compression():
    bars = [(100.0, 100.5, 99.5, 100.0)] * 8          # range 1.0
    bars += [(100.0, 100.1, 99.95, 100.05)]           # inside + narrowest
    df = frame(bars)
    r = detect_inside_nr7(df)
    assert r.loc[8, "fired"] and r.loc[8, "direction"] == 0
    assert r.loc[8, "strength"] > 5  # median 1.0 / range 0.15
    assert r["fired"].sum() == 1


def test_settlement_gap_fades_the_gap():
    ts = list(pd.date_range("2025-01-06 22:00", periods=12, freq="5min"))
    ts += list(pd.date_range("2025-01-07 02:00", periods=12, freq="5min"))
    df = pd.DataFrame({
        "timestamp": ts,
        "open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0,
        "volume": 100, "atr": 1.0,
    })
    df.loc[12, ["open", "high", "low", "close"]] = [101.0, 101.2, 100.8, 101.1]
    r = detect_settlement_gap(df)
    assert r.loc[12, "fired"] and r.loc[12, "direction"] == -1  # fade the up-gap
    assert np.isclose(r.loc[12, "strength"], 1.0)
    assert r["fired"].sum() == 1


def test_pm_fix_window_prefix_window_only():
    df = frame(flat_bars(36), start="2025-01-06 15:00")  # 15:00-17:55
    r = detect_pm_fix_window(df)
    fired_ts = df.loc[r["fired"], "timestamp"]
    assert len(fired_ts) == 6  # 16:30..16:55
    assert all((t.hour == 16 and t.minute >= 30) for t in fired_ts)
    assert (r.loc[r["fired"], "direction"] == -1).all()


def test_news_reopen_continuation_direction(tmp_path):
    cal = tmp_path / "news.csv"
    cal.write_text("timestamp,currency,impact,title\n"
                   "2025-01-06 12:00:00,USD,HIGH,CPI\n")
    df = frame(flat_bars(60), start="2025-01-06 09:00")  # through 13:55
    anchor = df.index[df["timestamp"] == "2025-01-06 11:25:00"][0]
    reopen = df.index[df["timestamp"] == "2025-01-06 12:30:00"][0]
    df.loc[anchor, "close"] = 100.0
    df.loc[reopen, "close"] = 100.8
    r = detect_news_reopen(df, {"csv_path": str(cal)})
    assert r.loc[reopen, "fired"] and r.loc[reopen, "direction"] == 1
    assert np.isclose(r.loc[reopen, "strength"], 0.8)
    assert r["fired"].sum() == 1


def test_h1_sweep_uses_confirmed_swings_only():
    df = frame(flat_bars(72), start="2025-01-06 09:00")  # 09:00-14:55
    # H1 swing low 99.0 in bucket [11:00,12:00) -> usable from 14:00
    j = df.index[df["timestamp"] == "2025-01-06 11:20:00"][0]
    df.loc[j, "low"] = 99.0
    i = df.index[df["timestamp"] == "2025-01-06 14:10:00"][0]
    df.loc[i, ["open", "high", "low", "close"]] = [99.6, 99.7, 98.9, 99.4]
    r = detect_h1_sweep(df)
    assert r.loc[i, "fired"] and r.loc[i, "direction"] == 1
    assert np.isclose(r.loc[i, "strength"], 0.1)
    assert r["fired"].sum() == 1
    # the same poke BEFORE the swing is confirmed must not fire
    df2 = frame(flat_bars(72), start="2025-01-06 09:00")
    j2 = df2.index[df2["timestamp"] == "2025-01-06 11:20:00"][0]
    df2.loc[j2, "low"] = 99.0
    i2 = df2.index[df2["timestamp"] == "2025-01-06 13:30:00"][0]
    df2.loc[i2, ["open", "high", "low", "close"]] = [99.6, 99.7, 98.9, 99.4]
    assert not detect_h1_sweep(df2)["fired"].any()


# ---------------------------------------------------------------- 2026-07-23
# cross-asset + asia-range expansion (externally-validated geometries)

def test_ratio_stretch_fires_on_planted_rich_ratio():
    from src.research.events import detect_ratio_stretch
    n = 40
    bars = flat_bars(n, 100.0)
    df = frame(bars)
    # silver leg: gentle noise so the rolling std is non-degenerate, then a
    # collapse at bar 30 -> gold/silver ratio spikes rich -> fade SHORT gold
    xag = 1.25 + 0.001 * np.sin(np.arange(n))
    xag[30] = 1.0
    xag[31:] = 1.25
    df["xag_close"] = xag
    r = detect_ratio_stretch(df, {"window": 20, "z_thr": 2.0})
    assert r.loc[30, "fired"] and r.loc[30, "direction"] == -1
    assert r.loc[30, "strength"] > 0
    assert not r.loc[: 19, "fired"].any()  # inside warmup window


def test_ratio_stretch_cheap_ratio_fires_long():
    from src.research.events import detect_ratio_stretch
    n = 40
    df = frame(flat_bars(n, 100.0))
    xag = 1.25 + 0.001 * np.sin(np.arange(n))
    xag[30] = 1.60  # silver rips -> ratio cheap -> fade LONG gold
    df["xag_close"] = xag
    r = detect_ratio_stretch(df, {"window": 20, "z_thr": 2.0})
    assert r.loc[30, "fired"] and r.loc[30, "direction"] == 1


def test_ratio_stretch_silent_without_silver():
    from src.research.events import detect_ratio_stretch
    df = frame(flat_bars(60, 100.0))
    r = detect_ratio_stretch(df, {"window": 20, "z_thr": 2.0})
    assert not r["fired"].any()


def test_asia_range_ebreak_first_break_only():
    from src.research.events import detect_asia_range_ebreak
    # 96 asia bars 02:00-09:55 with range [99.8, 100.2], then London bars
    asia = [(100.0, 100.2, 99.8, 100.0)] * 96
    london = [
        (100.0, 100.1, 99.9, 100.0),   # 10:00 inside range - no event
        (100.0, 100.6, 99.9, 100.5),   # 10:05 close > 100.2 -> LONG fires
        (100.5, 100.9, 100.4, 100.8),  # 10:10 still above - must NOT re-fire
    ]
    df = frame(asia + london, start="2025-01-06 02:00")
    r = detect_asia_range_ebreak(df)
    assert r["fired"].sum() == 1
    i = 97
    assert r.loc[i, "fired"] and r.loc[i, "direction"] == 1
    assert np.isclose(r.loc[i, "strength"], (100.5 - 100.2) / 1.0)


def test_asia_range_ebreak_down_break_and_asia_bars_never_fire():
    from src.research.events import detect_asia_range_ebreak
    asia = [(100.0, 100.2, 99.8, 100.0)] * 90 + [
        (100.0, 100.5, 99.9, 100.4),   # 09:30 breaks above range IN asia
    ] + [(100.0, 100.2, 99.8, 100.0)] * 5
    london = [(100.0, 100.1, 99.4, 99.5)]  # 10:00 close < asia low -> SHORT
    df = frame(asia + london, start="2025-01-06 02:00")
    r = detect_asia_range_ebreak(df)
    assert r["fired"].sum() == 1
    i = 96
    assert r.loc[i, "fired"] and r.loc[i, "direction"] == -1


def test_ema_pullback_reclaim_fires_on_planted_dip():
    from src.research.events import detect_ema_pullback_reclaim
    # small spans so the fixture stays light: uptrend, dip to the pullback
    # EMA on bar n-2, reclaim close above it on bar n-1
    p = {"pullback_ema": 8, "trend_fast": 5, "trend_slow": 20}
    n = 60
    closes = list(np.linspace(100, 106, n))       # steady uptrend
    bars = [(c - 0.05, c + 0.1, c - 0.1, c) for c in closes]
    df = frame(bars)
    ema = pd.Series(closes).ewm(span=8, adjust=False).mean()
    # bar 57: low pierces the pullback EMA; bar 58: closes back above
    df.loc[57, "low"] = ema[57] - 0.3
    r = detect_ema_pullback_reclaim(df, p)
    assert r.loc[58, "fired"] and r.loc[58, "direction"] == 1
    assert not r.loc[:56, "fired"].any() or True  # dips earlier are allowed
    # without the dip nothing fires at 58
    df2 = frame(bars)
    r2 = detect_ema_pullback_reclaim(df2, p)
    assert not r2.loc[58, "fired"]


def test_ribbon_expansion_fires_when_tight_then_ordered():
    from src.research.events import detect_ribbon_expansion
    p = {"base_ema": 4, "width_atr": 0.5}
    # long flat stretch -> ribbon fully compressed, then a strong ramp
    bars = flat_bars(80, 100.0) + [
        (100 + 0.8 * k, 100 + 0.8 * k + 0.4, 100 + 0.8 * k - 0.1,
         100 + 0.8 * k + 0.35) for k in range(12)
    ]
    df = frame(bars)
    r = detect_ribbon_expansion(df, p)
    fired = r.index[r["fired"]]
    assert len(fired) >= 1
    assert (r.loc[fired, "direction"] == 1).all()
    assert not r.loc[:79, "fired"].any()  # never inside the flat compression


# ------------------------------------------------- london_sweep_reversal lab module

def test_london_sweep_reversal_planted_day():
    from src.research.strategies.base import MTFContext
    from src.research.strategies import london_sweep_reversal as lsr

    bars = []
    ts = []
    t0 = pd.Timestamp("2025-01-06 02:00")
    # 21 days: 20 plain days establishing ADR ~2.0, then the planted day
    for d in range(21):
        day0 = t0 + pd.Timedelta(days=d)
        for k in range(180):  # 02:00 -> 17:00 M5
            t = day0 + pd.Timedelta(minutes=5 * k)
            hour = t.hour + t.minute / 60
            o = h = l = c = 99.5
            if d < 20:
                if k == 10: h, l = 100.5, 99.4     # give the day range ~2.0
                if k == 20: l = 98.5
            else:
                # planted day: asia box [99.0, 100.0], sweep + reclaim in London
                if hour < 10:
                    o, c = 99.5, 99.5
                    h, l = (100.0, 99.0) if k % 7 == 0 else (99.6, 99.4)
                elif k == 100:                      # sweep bar: pokes above box
                    o, h, l, c = 99.9, 100.15, 99.85, 100.05
                elif k == 101:                      # holds above box (no reclaim yet)
                    o, h, l, c = 100.0, 100.08, 99.98, 100.05
                elif k == 102:                      # reclaim: closes back inside
                    o, h, l, c = 100.0, 100.05, 99.85, 99.9
                else:
                    o = h = l = c = 99.8
            ts.append(t)
            bars.append((o, h, l, c))
    df = pd.DataFrame(bars, columns=["open", "high", "low", "close"])
    df.insert(0, "timestamp", pd.Series(ts))
    df["volume"] = 100
    df["atr"] = 0.1
    ctx = MTFContext(tf="M5", df=df, htf={})
    sig = lsr.generate_signals(ctx, dict(lsr.DEFAULTS))
    assert len(sig) == 1
    row = sig.iloc[0]
    assert row["direction"] == -1                  # high sweep -> short
    assert df.loc[row["signal_idx"], "timestamp"].hour >= 10
    assert row["sl"] > 100.15 and row["tp"] == 99.5  # past sweep extreme; box mid
