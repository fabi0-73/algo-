"""
Per-bar event detectors: the AMD strategy decomposed into atoms, each firing
hundreds-to-thousands of times over ~100k M5 bars so standalone edge can be
measured statistically (vs. the full AMD funnel's ~10 trades/month).

Contract: detect_<name>(df, params=None) -> DataFrame aligned to df.index with
    fired: bool      event COMPLETES at bar i (knowable at bar-i close)
    direction: int8  +1 long / -1 short hypothesis, 0 = no directional prior
    strength: float  size of the event in ATR units (0 when fired is False)

df is a prepare_frame() output: timestamp/open/high/low/close/volume/atr,
positional RangeIndex. NO LOOKAHEAD: row i may depend only on rows <= i
(tests enforce prefix-invariance mechanically). Stateful detectors use causal
O(n) loops; window detectors use vectorized rolling ops — both are fine, the
per-candle O(n*w) scan-backs of the AMD engine are what we avoid.

The loop-based originals in src/strategy/{fvg,order_blocks,market_structure}.py
define the pattern semantics and serve as test oracles; they are not called here.

HORIZONS is the fixed forward-return horizon set for ALL studies — chosen
upfront, never per-event (anti-horizon-shopping guardrail).
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .strategies.base import anchored_vwap, prior_day_stats, session_mask

HORIZONS = (1, 3, 6, 12, 24, 48)

# Broker-time sessions (~NY+7, see strategies/base.py header).
SESSIONS = {
    "asia": ("02:00", "10:00"),
    "london": ("10:00", "16:30"),
    "ny": ("16:30", "23:59"),
}


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "fired": np.zeros(n, dtype=bool),
        "direction": np.zeros(n, dtype=np.int8),
        "strength": np.zeros(n, dtype=float),
    })


def _merged(defaults: dict, params: dict = None) -> dict:
    out = dict(defaults)
    if params:
        out.update(params)
    return out


# ---------------------------------------------------------------- sweeps

SWEEP_DEFAULTS = {"lookback": 24, "min_poke_atr": 0.05}


def detect_sweep_high(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Wick beyond the prior `lookback`-bar high, close back below it:
    liquidity taken above the range -> reversal-short hypothesis."""
    p = _merged(SWEEP_DEFAULTS, params)
    out = _frame(len(df))
    prior_max = df["high"].rolling(p["lookback"]).max().shift(1)
    poke = (df["high"] - prior_max) / df["atr"]
    fired = (poke >= p["min_poke_atr"]) & (df["close"] < prior_max)
    fired &= prior_max.notna() & df["atr"].notna()
    out.loc[fired, "fired"] = True
    out.loc[fired, "direction"] = -1
    out.loc[fired, "strength"] = poke[fired]
    return out


def detect_sweep_low(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Mirror of detect_sweep_high: sweep below the range -> long hypothesis."""
    p = _merged(SWEEP_DEFAULTS, params)
    out = _frame(len(df))
    prior_min = df["low"].rolling(p["lookback"]).min().shift(1)
    poke = (prior_min - df["low"]) / df["atr"]
    fired = (poke >= p["min_poke_atr"]) & (df["close"] > prior_min)
    fired &= prior_min.notna() & df["atr"].notna()
    out.loc[fired, "fired"] = True
    out.loc[fired, "direction"] = 1
    out.loc[fired, "strength"] = poke[fired]
    return out


EQUAL_SWEEP_DEFAULTS = {"lookback": 48, "tol_atr": 0.10, "min_touches": 2,
                        "min_poke_atr": 0.05}


def _window_max_touches(values: np.ndarray, lookback: int, tol: np.ndarray):
    """For each bar i: max of values[i-lookback..i-1] and how many bars in
    that window sit within tol[i] of it. NaN/0 for i < lookback."""
    n = len(values)
    wmax = np.full(n, np.nan)
    touches = np.zeros(n, dtype=int)
    if n <= lookback:
        return wmax, touches
    win = sliding_window_view(values, lookback)      # win[k] = values[k:k+lookback]
    m = win.max(axis=1)                               # window ending at k+lookback-1
    wmax[lookback:] = m[: n - lookback]
    t = tol[lookback:]
    touches[lookback:] = (win[: n - lookback] >= (m[: n - lookback] - t)[:, None]).sum(axis=1)
    return wmax, touches


def detect_equal_level_sweep(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Sweep of EQUAL highs/lows (>= min_touches within tol_atr of the window
    extreme) — the engineered-liquidity version of a plain sweep."""
    p = _merged(EQUAL_SWEEP_DEFAULTS, params)
    out = _frame(len(df))
    atr = df["atr"].to_numpy(float)
    tol = p["tol_atr"] * atr

    hmax, htouch = _window_max_touches(df["high"].to_numpy(float), p["lookback"], tol)
    poke_hi = (df["high"].to_numpy(float) - hmax) / atr
    hi = (poke_hi >= p["min_poke_atr"]) & (df["close"].to_numpy(float) < hmax) \
        & (htouch >= p["min_touches"])

    lmin_neg, ltouch = _window_max_touches(-df["low"].to_numpy(float), p["lookback"], tol)
    lmin = -lmin_neg
    poke_lo = (lmin - df["low"].to_numpy(float)) / atr
    lo = (poke_lo >= p["min_poke_atr"]) & (df["close"].to_numpy(float) > lmin) \
        & (ltouch >= p["min_touches"])

    valid = ~np.isnan(atr)
    hi &= valid & ~np.isnan(hmax)
    lo &= valid & ~np.isnan(lmin)
    out.loc[hi, "fired"] = True
    out.loc[hi, "direction"] = -1
    out.loc[hi, "strength"] = poke_hi[hi]
    out.loc[lo & ~hi, "fired"] = True
    out.loc[lo & ~hi, "direction"] = 1
    out.loc[lo & ~hi, "strength"] = poke_lo[lo & ~hi]
    return out


# ---------------------------------------------------------------- FVG

FVG_DEFAULTS = {"min_size_atr": 0.10}  # matches STRATEGY.fvg_min_size_atr_mult


def detect_fvg_bull(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """3-bar bullish Fair Value Gap: high[i-2] < low[i] (src/strategy/fvg.py
    semantics). Fires at the third bar; strength = gap size in ATR."""
    p = _merged(FVG_DEFAULTS, params)
    out = _frame(len(df))
    gap = (df["low"] - df["high"].shift(2)) / df["atr"]
    fired = (gap >= p["min_size_atr"]) & df["atr"].notna()
    out.loc[fired, "fired"] = True
    out.loc[fired, "direction"] = 1
    out.loc[fired, "strength"] = gap[fired]
    return out


def detect_fvg_bear(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """3-bar bearish FVG: low[i-2] > high[i]."""
    p = _merged(FVG_DEFAULTS, params)
    out = _frame(len(df))
    gap = (df["low"].shift(2) - df["high"]) / df["atr"]
    fired = (gap >= p["min_size_atr"]) & df["atr"].notna()
    out.loc[fired, "fired"] = True
    out.loc[fired, "direction"] = -1
    out.loc[fired, "strength"] = gap[fired]
    return out


FVG_FILL_DEFAULTS = {"min_size_atr": 0.10, "max_age_bars": 72}


def detect_fvg_fill_bull(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """First retrace INTO the most recent unfilled bullish FVG within
    max_age_bars: price returns to the gap -> long-continuation hypothesis.
    Causal O(n) state loop (one active gap at a time, newest wins)."""
    p = _merged(FVG_FILL_DEFAULTS, params)
    return _fvg_fill(df, p, bull=True)


def detect_fvg_fill_bear(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Mirror: first retrace up into the most recent bearish FVG."""
    p = _merged(FVG_FILL_DEFAULTS, params)
    return _fvg_fill(df, p, bull=False)


def _fvg_fill(df: pd.DataFrame, p: dict, bull: bool) -> pd.DataFrame:
    out = _frame(len(df))
    creator = detect_fvg_bull(df, p) if bull else detect_fvg_bear(df, p)
    created = creator["fired"].to_numpy()
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    fired = np.zeros(len(df), dtype=bool)
    strength = np.zeros(len(df))
    top = bottom = None
    born = -1
    for i in range(len(df)):
        if top is not None and i - born <= p["max_age_bars"]:
            if bull and low[i] <= top:
                fired[i] = True
                strength[i] = (top - low[i]) / atr[i] if atr[i] > 0 else 0.0
                top = bottom = None
            elif not bull and high[i] >= bottom:
                fired[i] = True
                strength[i] = (high[i] - bottom) / atr[i] if atr[i] > 0 else 0.0
                top = bottom = None
        elif top is not None:
            top = bottom = None  # expired
        if created[i]:
            # gap zone: bull [high[i-2], low[i]], bear [high[i], low[i-2]]
            if bull:
                bottom, top = high[i - 2], low[i]
            else:
                bottom, top = high[i], low[i - 2]
            born = i
    out["fired"] = fired
    out.loc[fired, "direction"] = 1 if bull else -1
    out["strength"] = strength
    return out


# ---------------------------------------------------------------- structure

BOS_DEFAULTS = {"swing_strength": 3}


def detect_bos_up(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Break of structure up: first close above the most recent CONFIRMED
    swing high (fractal of `swing_strength`, confirmed swing_strength bars
    after the pivot). Fires at the confirmation/break bar, never the pivot.
    Continuation-long hypothesis. One fire per swing level."""
    return _bos(df, _merged(BOS_DEFAULTS, params), up=True)


def detect_bos_down(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Mirror: first close below the most recent confirmed swing low."""
    return _bos(df, _merged(BOS_DEFAULTS, params), up=False)


def _bos(df: pd.DataFrame, p: dict, up: bool) -> pd.DataFrame:
    out = _frame(len(df))
    k = p["swing_strength"]
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    n = len(df)
    fired = np.zeros(n, dtype=bool)
    strength = np.zeros(n)
    level = None
    for i in range(n):
        # a pivot at j = i-k is confirmed at bar i (k bars each side)
        j = i - k
        if j >= k:
            if up and high[j] == high[j - k: j + k + 1].max() \
                    and (high[j] > high[j - k: j]).all() and (high[j] > high[j + 1: j + k + 1]).all():
                level = high[j]
            if not up and low[j] == low[j - k: j + k + 1].min() \
                    and (low[j] < low[j - k: j]).all() and (low[j] < low[j + 1: j + k + 1]).all():
                level = low[j]
        if level is not None:
            if up and close[i] > level:
                fired[i] = True
                strength[i] = (close[i] - level) / atr[i] if atr[i] > 0 else 0.0
                level = None
            elif not up and close[i] < level:
                fired[i] = True
                strength[i] = (level - close[i]) / atr[i] if atr[i] > 0 else 0.0
                level = None
    out["fired"] = fired
    out.loc[fired, "direction"] = 1 if up else -1
    out["strength"] = strength
    return out


# ---------------------------------------------------------------- displacement / OB

DISPLACEMENT_DEFAULTS = {"min_body_atr": 1.5, "min_body_frac": 0.7}


def detect_displacement(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Single-candle displacement: body >= min_body_atr ATR and body is at
    least min_body_frac of the bar range. Direction = sign of the body."""
    p = _merged(DISPLACEMENT_DEFAULTS, params)
    out = _frame(len(df))
    body = df["close"] - df["open"]
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body_atr = body.abs() / df["atr"]
    fired = (body_atr >= p["min_body_atr"]) & (body.abs() / rng >= p["min_body_frac"])
    fired &= df["atr"].notna() & fired.notna()
    fired = fired.fillna(False)
    out.loc[fired, "fired"] = True
    out.loc[fired, "direction"] = np.sign(body[fired]).astype(np.int8)
    out.loc[fired, "strength"] = body_atr[fired]
    return out


OB_RETEST_DEFAULTS = {"min_body_atr": 1.5, "min_body_frac": 0.7, "max_age_bars": 72}


def detect_ob_retest_bull(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Bullish order block = last down candle before an up displacement
    (src/strategy/order_blocks.py semantics). Fires on the first later bar
    that trades back into the OB zone [low, high] of that candle."""
    return _ob_retest(df, _merged(OB_RETEST_DEFAULTS, params), bull=True)


def detect_ob_retest_bear(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Mirror: last up candle before a down displacement, retested from below."""
    return _ob_retest(df, _merged(OB_RETEST_DEFAULTS, params), bull=False)


def _ob_retest(df: pd.DataFrame, p: dict, bull: bool) -> pd.DataFrame:
    out = _frame(len(df))
    disp = detect_displacement(df, p)
    disp_fired = (disp["fired"] & (disp["direction"] == (1 if bull else -1))).to_numpy()
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    n = len(df)
    fired = np.zeros(n, dtype=bool)
    strength = np.zeros(n)
    zone = None  # (bottom, top)
    born = -1
    for i in range(n):
        if zone is not None and i - born <= p["max_age_bars"]:
            bottom, top = zone
            if bull and low[i] <= top and i > born:
                fired[i] = True
                strength[i] = (top - low[i]) / atr[i] if atr[i] > 0 else 0.0
                zone = None
            elif not bull and high[i] >= bottom and i > born:
                fired[i] = True
                strength[i] = (high[i] - bottom) / atr[i] if atr[i] > 0 else 0.0
                zone = None
        elif zone is not None:
            zone = None
        if disp_fired[i]:
            # walk back to the last opposite-color candle (bounded lookback 10)
            for j in range(i - 1, max(i - 11, -1), -1):
                if bull and c[j] < o[j]:
                    zone = (low[j], high[j])
                    born = i
                    break
                if not bull and c[j] > o[j]:
                    zone = (low[j], high[j])
                    born = i
                    break
    out["fired"] = fired
    out.loc[fired, "direction"] = 1 if bull else -1
    out["strength"] = strength
    return out


# ---------------------------------------------------------------- range break

RANGE_BREAK_DEFAULTS = {"window": 36, "max_width_atr": 3.0}


def detect_range_break(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Close beyond a compressed prior range: rolling `window`-bar range with
    width <= max_width_atr ATR (consolidation.py's tightness idea), first
    close outside. Breakout-continuation hypothesis."""
    p = _merged(RANGE_BREAK_DEFAULTS, params)
    out = _frame(len(df))
    hi = df["high"].rolling(p["window"]).max().shift(1)
    lo = df["low"].rolling(p["window"]).min().shift(1)
    width = (hi - lo) / df["atr"]
    tight = width <= p["max_width_atr"]
    up = tight & (df["close"] > hi)
    dn = tight & (df["close"] < lo) & ~up
    valid = hi.notna() & df["atr"].notna()
    up &= valid
    dn &= valid
    out.loc[up, "fired"] = True
    out.loc[up, "direction"] = 1
    out.loc[up, "strength"] = ((df["close"] - hi) / df["atr"])[up]
    out.loc[dn, "fired"] = True
    out.loc[dn, "direction"] = -1
    out.loc[dn, "strength"] = ((lo - df["close"]) / df["atr"])[dn]
    return out


# ---------------------------------------------------------------- sessions / judas

SESSION_OPEN_DEFAULTS = {"session": "london"}


def _session_open(df: pd.DataFrame, session: str) -> pd.DataFrame:
    out = _frame(len(df))
    start, end = SESSIONS[session]
    in_sess = session_mask(df["timestamp"], start, end)
    day = df["timestamp"].dt.normalize()
    # first bar of the session each day: in-session and previous bar of the
    # same day was not yet in-session
    prev_in = in_sess.shift(1, fill_value=False) & (day == day.shift(1))
    first = in_sess & ~prev_in
    out.loc[first, "fired"] = True
    return out


def detect_session_open_asia(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """First bar of the Asia session (broker time). No directional prior."""
    return _session_open(df, "asia")


def detect_session_open_london(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """First bar of the London session (broker time)."""
    return _session_open(df, "london")


def detect_session_open_ny(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """First bar of the NY session (broker time)."""
    return _session_open(df, "ny")


JUDAS_DEFAULTS = {"lookback": 24, "min_poke_atr": 0.05, "window_min": 90}


def detect_judas(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Judas swing, decomposed: a sweep (either side) inside the first
    `window_min` minutes after London open — the fast fake move that the AMD
    manipulation phase looks for (manipulation.py + Judas quality gate)."""
    p = _merged(JUDAS_DEFAULTS, params)
    hi = detect_sweep_high(df, p)
    lo = detect_sweep_low(df, p)
    start, _ = SESSIONS["london"]
    sh, sm = (int(x) for x in start.split(":"))
    start_min = sh * 60 + sm
    mod = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    in_window = (mod >= start_min) & (mod < start_min + p["window_min"])
    out = _frame(len(df))
    any_sweep = (hi["fired"] | lo["fired"]) & in_window
    out.loc[any_sweep, "fired"] = True
    out.loc[any_sweep, "direction"] = np.where(
        hi["fired"][any_sweep], -1, 1).astype(np.int8)
    out.loc[any_sweep, "strength"] = np.where(
        hi["fired"][any_sweep], hi["strength"][any_sweep], lo["strength"][any_sweep])
    return out


# ---------------------------------------------------------------- volume / levels / vwap

VOL_SPIKE_DEFAULTS = {"window": 48, "min_mult": 3.0}


def detect_vol_spike(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Tick volume >= min_mult x rolling median (liquidity proxy only).
    No directional prior; strength = multiple of median."""
    p = _merged(VOL_SPIKE_DEFAULTS, params)
    out = _frame(len(df))
    med = df["volume"].rolling(p["window"]).median().shift(1)
    mult = df["volume"] / med.replace(0, np.nan)
    fired = (mult >= p["min_mult"]) & med.notna()
    fired = fired.fillna(False)
    out.loc[fired, "fired"] = True
    out.loc[fired, "strength"] = mult[fired]
    return out


PD_LEVEL_DEFAULTS = {}


def detect_pdh_touch(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Touch + rejection at the prior day high: high pokes PDH, close back
    below -> reversal-short hypothesis (key_levels.py PDH atom)."""
    out = _frame(len(df))
    pd_stats = prior_day_stats(df)
    lvl = pd_stats["d_high"]
    fired = (df["high"] >= lvl) & (df["close"] < lvl) & lvl.notna() & df["atr"].notna()
    out.loc[fired, "fired"] = True
    out.loc[fired, "direction"] = -1
    out.loc[fired, "strength"] = ((df["high"] - lvl) / df["atr"])[fired]
    return out


def detect_pdl_touch(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Mirror: touch + rejection at the prior day low -> long hypothesis."""
    out = _frame(len(df))
    pd_stats = prior_day_stats(df)
    lvl = pd_stats["d_low"]
    fired = (df["low"] <= lvl) & (df["close"] > lvl) & lvl.notna() & df["atr"].notna()
    out.loc[fired, "fired"] = True
    out.loc[fired, "direction"] = 1
    out.loc[fired, "strength"] = ((lvl - df["low"]) / df["atr"])[fired]
    return out


def detect_pdh_break(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """First close of the day above the prior day high -> continuation-long."""
    out = _frame(len(df))
    pd_stats = prior_day_stats(df)
    lvl = pd_stats["d_high"]
    above = (df["close"] > lvl) & lvl.notna() & df["atr"].notna()
    day = df["timestamp"].dt.normalize()
    first = above & ~above.groupby(day).cummax().shift(1, fill_value=False).where(day == day.shift(1), False)
    out.loc[first, "fired"] = True
    out.loc[first, "direction"] = 1
    out.loc[first, "strength"] = ((df["close"] - lvl) / df["atr"])[first]
    return out


def detect_pdl_break(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Mirror: first close of the day below the prior day low."""
    out = _frame(len(df))
    pd_stats = prior_day_stats(df)
    lvl = pd_stats["d_low"]
    below = (df["close"] < lvl) & lvl.notna() & df["atr"].notna()
    day = df["timestamp"].dt.normalize()
    first = below & ~below.groupby(day).cummax().shift(1, fill_value=False).where(day == day.shift(1), False)
    out.loc[first, "fired"] = True
    out.loc[first, "direction"] = -1
    out.loc[first, "strength"] = ((lvl - df["close"]) / df["atr"])[first]
    return out


VWAP_STRETCH_DEFAULTS = {"min_stretch_atr": 1.5}


def detect_vwap_stretch(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Close stretched >= min_stretch_atr ATR from the daily anchored VWAP ->
    mean-reversion hypothesis toward VWAP."""
    p = _merged(VWAP_STRETCH_DEFAULTS, params)
    out = _frame(len(df))
    vwap = anchored_vwap(df)
    stretch = (df["close"] - vwap) / df["atr"]
    fired = (stretch.abs() >= p["min_stretch_atr"]) & vwap.notna() & df["atr"].notna()
    fired = fired.fillna(False)
    out.loc[fired, "fired"] = True
    out.loc[fired, "direction"] = (-np.sign(stretch[fired])).astype(np.int8)
    out.loc[fired, "strength"] = stretch.abs()[fired]
    return out


# ---------------------------------------------------------------- opening range

# NY or_minutes=60 matches the validated NY_IB initial balance (16:30-17:30);
# London 30min is the classic ORB window after the 10:00 broker open.
ORB_LDN_DEFAULTS = {"session": "london", "or_minutes": 30}
ORB_NY_DEFAULTS = {"session": "ny", "or_minutes": 60}


def _orb_levels(df: pd.DataFrame, session: str, or_minutes: int):
    """Frozen opening-range high/low per day: extremes of the first
    `or_minutes` of the session, NaN until that window has fully closed —
    causal by construction (only same-day past bars enter the cummax)."""
    start, _ = SESSIONS[session]
    sh, sm = (int(x) for x in start.split(":"))
    start_min = sh * 60 + sm
    mod = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    day = df["timestamp"].dt.normalize()
    in_or = (mod >= start_min) & (mod < start_min + or_minutes)
    after = mod >= start_min + or_minutes
    or_hi = df["high"].where(in_or).groupby(day).cummax().groupby(day).ffill().where(after)
    or_lo = df["low"].where(in_or).groupby(day).cummin().groupby(day).ffill().where(after)
    return or_hi, or_lo, day


def _orb_break(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    out = _frame(len(df))
    or_hi, or_lo, day = _orb_levels(df, p["session"], p["or_minutes"])
    atr = df["atr"]
    same_day = day == day.shift(1)
    up = (df["close"] > or_hi) & or_hi.notna() & atr.notna()
    dn = (df["close"] < or_lo) & or_lo.notna() & atr.notna()
    first_up = up & ~up.groupby(day).cummax().shift(1, fill_value=False).where(same_day, False)
    first_dn = dn & ~dn.groupby(day).cummax().shift(1, fill_value=False).where(same_day, False)
    first_dn &= ~first_up
    out.loc[first_up, "fired"] = True
    out.loc[first_up, "direction"] = 1
    out.loc[first_up, "strength"] = ((df["close"] - or_hi) / atr)[first_up]
    out.loc[first_dn, "fired"] = True
    out.loc[first_dn, "direction"] = -1
    out.loc[first_dn, "strength"] = ((or_lo - df["close"]) / atr)[first_dn]
    return out


def detect_orb_break_london(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """First close beyond the London opening range (per side, per day):
    breakout-continuation hypothesis. Miner-ready atomization of the ORB
    family (the only lab KEEPER, ny_ib, is this family's pullback variant)."""
    return _orb_break(df, _merged(ORB_LDN_DEFAULTS, params))


def detect_orb_break_ny(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """First close beyond the NY initial balance (16:30-17:30 broker)."""
    return _orb_break(df, _merged(ORB_NY_DEFAULTS, params))


ORB_PB_LDN_DEFAULTS = {"session": "london", "or_minutes": 30,
                       "tol_atr": 0.15, "max_age_bars": 24}
ORB_PB_NY_DEFAULTS = {"session": "ny", "or_minutes": 60,
                      "tol_atr": 0.15, "max_age_bars": 24}


def _orb_pullback(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """After an ORB break, the first bar that trades back to the broken edge
    (within tol_atr) and CLOSES in the break direction -> continuation. The
    retrace-to-broken-level geometry that separated the KEEPER (ny_ib) from
    the killed enter-at-the-break atoms. Causal O(n) state loop."""
    out = _frame(len(df))
    breaks = _orb_break(df, p)
    or_hi, or_lo, _ = _orb_levels(df, p["session"], p["or_minutes"])
    bf = breaks["fired"].to_numpy()
    bd = breaks["direction"].to_numpy()
    hi_arr = or_hi.to_numpy(float)
    lo_arr = or_lo.to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    n = len(df)
    fired = np.zeros(n, dtype=bool)
    direction = np.zeros(n, dtype=np.int8)
    strength = np.zeros(n)
    level, d, born = None, 0, -1
    for i in range(n):
        if level is not None and i - born > p["max_age_bars"]:
            level = None
        if level is not None and i > born and atr[i] > 0:
            tol = p["tol_atr"] * atr[i]
            if d > 0 and low[i] <= level + tol and close[i] > level:
                fired[i] = True
                direction[i] = 1
                strength[i] = (close[i] - level) / atr[i]
                level = None
            elif d < 0 and high[i] >= level - tol and close[i] < level:
                fired[i] = True
                direction[i] = -1
                strength[i] = (level - close[i]) / atr[i]
                level = None
        if bf[i]:
            d = int(bd[i])
            level = hi_arr[i] if d > 0 else lo_arr[i]
            born = i
    out["fired"] = fired
    out["direction"] = direction
    out["strength"] = strength
    return out


def detect_orb_pullback_london(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Retrace to the broken London OR edge, close in break direction."""
    return _orb_pullback(df, _merged(ORB_PB_LDN_DEFAULTS, params))


def detect_orb_pullback_ny(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Retrace to the broken NY IB edge, close in break direction."""
    return _orb_pullback(df, _merged(ORB_PB_NY_DEFAULTS, params))


# ---------------------------------------------------------------- confirmation / fade atoms

SWEEP_RECLAIM_DEFAULTS = {"lookback": 24, "min_poke_atr": 0.05, "confirm_bars": 6}


def detect_sweep_reclaim(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Sweep -> CONFIRMED reclaim: after a sweep_low at bar s, the first close
    above high[s] within confirm_bars -> long (mirror for sweep_high). Fires
    on the confirmation bar, never the poke — the manipulation->distribution
    handoff the AMD engine trades, as one atom. Distinct population from raw
    sweeps (only the confirmed subset, bars later)."""
    p = _merged(SWEEP_RECLAIM_DEFAULTS, params)
    lo = detect_sweep_low(df, p)["fired"].to_numpy()
    hi = detect_sweep_high(df, p)["fired"].to_numpy()
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    n = len(df)
    out = _frame(n)
    fired = np.zeros(n, dtype=bool)
    direction = np.zeros(n, dtype=np.int8)
    strength = np.zeros(n)
    d, trig, born = 0, None, -1
    for i in range(n):
        if trig is not None and i - born > p["confirm_bars"]:
            trig = None
        if trig is not None and i > born and atr[i] > 0:
            if d > 0 and close[i] > trig:
                fired[i] = True
                direction[i] = 1
                strength[i] = (close[i] - trig) / atr[i]
                trig = None
            elif d < 0 and close[i] < trig:
                fired[i] = True
                direction[i] = -1
                strength[i] = (trig - close[i]) / atr[i]
                trig = None
        # newest sweep wins (poke bar itself can never confirm)
        if lo[i]:
            d, trig, born = 1, high[i], i
        elif hi[i]:
            d, trig, born = -1, low[i], i
    out["fired"] = fired
    out["direction"] = direction
    out["strength"] = strength
    return out


FAILED_BREAK_DEFAULTS = {"window": 36, "max_width_atr": 3.0, "confirm_bars": 6}


def detect_failed_break(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Failed compressed-range breakout: a range_break whose price CLOSES back
    inside the old range within confirm_bars -> fade in the reversal
    direction (trapped breakout traders). The complement population of the
    killed break-continuation atoms."""
    p = _merged(FAILED_BREAK_DEFAULTS, params)
    breaks = detect_range_break(df, p)
    hi = df["high"].rolling(p["window"]).max().shift(1).to_numpy(float)
    lo = df["low"].rolling(p["window"]).min().shift(1).to_numpy(float)
    bf = breaks["fired"].to_numpy()
    bd = breaks["direction"].to_numpy()
    close = df["close"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    n = len(df)
    out = _frame(n)
    fired = np.zeros(n, dtype=bool)
    direction = np.zeros(n, dtype=np.int8)
    strength = np.zeros(n)
    d, edge, born = 0, None, -1
    for i in range(n):
        if edge is not None and i - born > p["confirm_bars"]:
            edge = None
        if edge is not None and i > born and atr[i] > 0:
            if d > 0 and close[i] < edge:
                fired[i] = True
                direction[i] = -1
                strength[i] = (edge - close[i]) / atr[i]
                edge = None
            elif d < 0 and close[i] > edge:
                fired[i] = True
                direction[i] = 1
                strength[i] = (close[i] - edge) / atr[i]
                edge = None
        if bf[i]:
            d = int(bd[i])
            edge = hi[i] if d > 0 else lo[i]  # the broken range edge
            born = i
    out["fired"] = fired
    out["direction"] = direction
    out["strength"] = strength
    return out


WICK_REJECT_DEFAULTS = {"min_wick_atr": 0.75, "max_body_frac": 0.35}


def detect_wick_rejection(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Pin bar: small body, one dominant wick, close in the opposite third of
    the range -> away-from-the-wick hypothesis. The INVERSE shape of
    displacement (small body / long wick vs big body), and unanchored — it
    can fire in open air where no sweep/level atom sees anything."""
    p = _merged(WICK_REJECT_DEFAULTS, params)
    out = _frame(len(df))
    o = df["open"]
    c = df["close"]
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = (c - o).abs()
    upper = df["high"] - np.maximum(o, c)
    lower = np.minimum(o, c) - df["low"]
    small_body = body <= p["max_body_frac"] * rng
    valid = df["atr"].notna() & rng.notna()
    lo_rej = small_body & valid & (lower / df["atr"] >= p["min_wick_atr"]) \
        & (c >= df["high"] - rng / 3)
    hi_rej = small_body & valid & (upper / df["atr"] >= p["min_wick_atr"]) \
        & (c <= df["low"] + rng / 3) & ~lo_rej
    lo_rej = lo_rej.fillna(False)
    hi_rej = hi_rej.fillna(False)
    out.loc[lo_rej, "fired"] = True
    out.loc[lo_rej, "direction"] = 1
    out.loc[lo_rej, "strength"] = (lower / df["atr"])[lo_rej]
    out.loc[hi_rej, "fired"] = True
    out.loc[hi_rej, "direction"] = -1
    out.loc[hi_rej, "strength"] = (upper / df["atr"])[hi_rej]
    return out


ROUND_LEVEL_DEFAULTS = {"grid_usd": 10.0, "min_poke_atr": 0.02}


def detect_round_level_reject(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Poke-and-reject at static round-number levels (every grid_usd): open
    below the next $10 handle, high pokes it, close back below -> short
    (mirror for support). Psychological-barrier literature (Aggarwal &
    Lucey); static grid = no lookahead possible. Distinct from pdh/pdl and
    VWAP: those are dynamic levels, this one never moves."""
    p = _merged(ROUND_LEVEL_DEFAULTS, params)
    out = _frame(len(df))
    g = p["grid_usd"]
    o = df["open"].to_numpy(float)
    up_lvl = np.ceil(o / g) * g
    dn_lvl = np.floor(o / g) * g
    atr = df["atr"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        poke_up = (high - up_lvl) / atr
        poke_dn = (dn_lvl - low) / atr
    valid = ~np.isnan(atr) & (atr > 0)
    short = valid & (o < up_lvl) & (poke_up >= p["min_poke_atr"]) & (close < up_lvl)
    long_ = valid & (o > dn_lvl) & (poke_dn >= p["min_poke_atr"]) & (close > dn_lvl) & ~short
    out.loc[short, "fired"] = True
    out.loc[short, "direction"] = -1
    out.loc[short, "strength"] = poke_up[short]
    out.loc[long_, "fired"] = True
    out.loc[long_, "direction"] = 1
    out.loc[long_, "strength"] = poke_dn[long_]
    return out


# ---------------------------------------------------------------- compression / gaps / time

VOL_DRYUP_DEFAULTS = {"window": 48, "max_frac": 0.4}


def detect_vol_dryup(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Participation dry-up: tick volume <= max_frac x rolling median — the
    opposite tail of vol_spike (climax) and the classic precursor of
    expansion. No directional prior; strength = median/volume multiple."""
    p = _merged(VOL_DRYUP_DEFAULTS, params)
    out = _frame(len(df))
    med = df["volume"].rolling(p["window"]).median().shift(1)
    vol = df["volume"].replace(0, np.nan)
    frac = vol / med.replace(0, np.nan)
    fired = (frac <= p["max_frac"]) & med.notna() & df["atr"].notna()
    fired = fired.fillna(False)
    out.loc[fired, "fired"] = True
    out.loc[fired, "strength"] = (1.0 / frac)[fired]
    return out


NR_DEFAULTS = {"nr_window": 7}


def detect_inside_nr7(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Inside bar that is also the narrowest range of the last nr_window bars
    (Crabel NR7 + inside day): anticipatory compression atom, fires BEFORE
    any breakout (range_break fires after). No directional prior; strength =
    prior median range / current range (compression ratio)."""
    p = _merged(NR_DEFAULTS, params)
    out = _frame(len(df))
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    inside = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    narrowest = rng <= rng.rolling(p["nr_window"]).min()
    med_prior = rng.shift(1).rolling(p["nr_window"] - 1).median()
    fired = inside & narrowest & med_prior.notna() & rng.notna() & df["atr"].notna()
    fired = fired.fillna(False)
    out.loc[fired, "fired"] = True
    out.loc[fired, "strength"] = (med_prior / rng)[fired]
    return out


GAP_DEFAULTS = {"min_gap_atr": 0.5}


def detect_settlement_gap(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Open gap on the first bar of a new trading day (weekend/settlement):
    fade hypothesis (gap-fill), direction = against the gap. Strength =
    |gap|/ATR. Uses the gap taxonomy the data audit already validates."""
    p = _merged(GAP_DEFAULTS, params)
    out = _frame(len(df))
    day = df["timestamp"].dt.normalize()
    new_day = (day != day.shift(1)) & df["close"].shift(1).notna()
    gap = (df["open"] - df["close"].shift(1)) / df["atr"]
    fired = new_day & (gap.abs() >= p["min_gap_atr"]) & df["atr"].notna()
    fired = fired.fillna(False)
    out.loc[fired, "fired"] = True
    out.loc[fired, "direction"] = (-np.sign(gap[fired])).astype(np.int8)
    out.loc[fired, "strength"] = gap.abs()[fired]
    return out


PM_FIX_DEFAULTS = {"start": "16:30", "end": "17:00"}


def detect_pm_fix_window(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """London PM gold fix approach window (15:00 London = 17:00 broker):
    documented systematic decline into the fix (Speck's 1-min study; LBMA
    debate) -> short during the pre-fix window. NOTE: pure time-of-day drift
    is nulled BY CONSTRUCTION in the TOD-matched excess (deliberate
    anti-seasonality discipline) — the informative statistics for this atom
    are the CI-vs-zero columns in the event study. Strength = VWAP stretch
    at the bar, so terciles carry a non-time dimension (stretched-above-VWAP
    into the fix is the strongest form of the hypothesis)."""
    p = _merged(PM_FIX_DEFAULTS, params)
    out = _frame(len(df))
    in_win = session_mask(df["timestamp"], p["start"], p["end"])
    vwap = anchored_vwap(df)
    stretch = ((df["close"] - vwap) / df["atr"]).clip(lower=0.0)
    fired = in_win & df["atr"].notna() & vwap.notna()
    fired = fired.fillna(False)
    out.loc[fired, "fired"] = True
    out.loc[fired, "direction"] = -1
    out.loc[fired, "strength"] = stretch[fired].fillna(0.0)
    return out


NEWS_REOPEN_DEFAULTS = {"csv_path": "data/news_events.csv",
                        "pre_minutes": 30, "post_minutes": 30}


def detect_news_reopen(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """First bar after a scheduled-news blackout window ends: direction =
    sign of the move ACROSS the blackout (post-news continuation, the
    post-announcement-drift literature). The calendar is scheduled macro
    releases (static file, broker frame — dates knowable in advance, so no
    lookahead); missing file -> no fires. Strength = |blackout move|/ATR."""
    p = _merged(NEWS_REOPEN_DEFAULTS, params)
    out = _frame(len(df))
    from pathlib import Path
    path = Path(p["csv_path"])
    if not path.exists() or len(df) == 0:
        return out
    try:
        cal = pd.read_csv(path)
        ev_ts = pd.to_datetime(cal["timestamp"]).sort_values().to_numpy()
    except Exception:
        return out
    ts = df["timestamp"].to_numpy()
    close = df["close"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    pre = np.timedelta64(p["pre_minutes"], "m")
    post = np.timedelta64(p["post_minutes"], "m")
    for t in ev_ts:
        anchor = np.searchsorted(ts, t - pre) - 1   # last bar before blackout
        reopen = np.searchsorted(ts, t + post)      # first bar at/after end
        if anchor < 0 or reopen >= len(df) or reopen <= anchor:
            continue
        if not (np.isfinite(atr[reopen]) and atr[reopen] > 0):
            continue
        move = close[reopen] - close[anchor]
        if move == 0:
            continue
        out.iloc[reopen, 0] = True                  # fired
        out.iloc[reopen, 1] = np.int8(1 if move > 0 else -1)
        out.iloc[reopen, 2] = abs(move) / atr[reopen]
    return out


H1_SWEEP_DEFAULTS = {"swing_strength": 2, "min_poke_atr": 0.05,
                     "max_swing_age_h1": 48}


def detect_h1_sweep(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Sweep-and-reject of a CONFIRMED H1 swing level, detected on M5: low
    pokes below the last confirmed H1 swing low, close back above -> long
    (mirror above H1 swing highs). Higher-timeframe structure is a different
    liquidity population from any M5 atom. Causality: an H1 swing at bucket
    j (fractal of swing_strength) is usable only from the m5 bar whose
    timestamp >= the END of bucket j+swing_strength — completed H1 data
    only, so prefix-invariance holds mechanically. One fire per swing."""
    p = _merged(H1_SWEEP_DEFAULTS, params)
    out = _frame(len(df))
    n = len(df)
    if n == 0:
        return out
    k = int(p["swing_strength"])
    h1 = (df.set_index("timestamp")
            .resample("1h", label="right", closed="left")
            .agg({"high": "max", "low": "min"})
            .dropna())
    if len(h1) < 2 * k + 1:
        return out
    h1_high = h1["high"].to_numpy(float)
    h1_low = h1["low"].to_numpy(float)
    h1_end = h1.index.to_numpy()  # label=right == bucket END time
    # confirmed swings: (usable_from_time, level); usable when the k-th
    # neighbor bucket has CLOSED
    swings_hi, swings_lo = [], []
    for j in range(k, len(h1) - k):
        seg_h = h1_high[j - k: j + k + 1]
        if h1_high[j] == seg_h.max() and (h1_high[j] > seg_h[:k]).all() \
                and (h1_high[j] > seg_h[k + 1:]).all():
            swings_hi.append((h1_end[j + k], h1_high[j]))
        seg_l = h1_low[j - k: j + k + 1]
        if h1_low[j] == seg_l.min() and (h1_low[j] < seg_l[:k]).all() \
                and (h1_low[j] < seg_l[k + 1:]).all():
            swings_lo.append((h1_end[j + k], h1_low[j]))
    ts = df["timestamp"].to_numpy()
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    fired = np.zeros(n, dtype=bool)
    direction = np.zeros(n, dtype=np.int8)
    strength = np.zeros(n)
    max_age = np.timedelta64(int(p["max_swing_age_h1"]), "h")
    pi = pj = 0
    cur_hi = cur_lo = None  # (usable_from, level)
    for i in range(n):
        while pi < len(swings_hi) and swings_hi[pi][0] <= ts[i]:
            cur_hi = swings_hi[pi]
            pi += 1
        while pj < len(swings_lo) and swings_lo[pj][0] <= ts[i]:
            cur_lo = swings_lo[pj]
            pj += 1
        if cur_hi is not None and ts[i] - cur_hi[0] > max_age:
            cur_hi = None
        if cur_lo is not None and ts[i] - cur_lo[0] > max_age:
            cur_lo = None
        if atr[i] <= 0 or not np.isfinite(atr[i]):
            continue
        if cur_lo is not None:
            poke = (cur_lo[1] - low[i]) / atr[i]
            if poke >= p["min_poke_atr"] and close[i] > cur_lo[1]:
                fired[i] = True
                direction[i] = 1
                strength[i] = poke
                cur_lo = None
        if not fired[i] and cur_hi is not None:
            poke = (high[i] - cur_hi[1]) / atr[i]
            if poke >= p["min_poke_atr"] and close[i] < cur_hi[1]:
                fired[i] = True
                direction[i] = -1
                strength[i] = poke
                cur_hi = None
    out["fired"] = fired
    out["direction"] = direction
    out["strength"] = strength
    return out


# ------------------------------------------------- cross-asset / asia range
# 2026-07-23 expansion. Idea provenance: the two most durable high-count M5
# streams in the (independently analyzed) quant-trading-lab repo — gold/silver
# ratio z-stretch fade (their WFO chose fade over follow) and the Asia-range
# early break. Geometry re-implemented for OUR data contract and judged by OUR
# gauntlet; nothing is imported as evidence. Silver caveat: broker retains
# ~15-17mo of XAG M5 (cache starts 2025-01-31), so train-window n is thin —
# treat event-study output as indicative until history accumulates.

RATIO_STRETCH_DEFAULTS = {"window": 440, "z_thr": 2.0}


def attach_xag(df: pd.DataFrame, path=None, max_gap_bars: int = 3) -> pd.DataFrame:
    """Left-merge the cached XAGUSD M5 close onto df as `xag_close`.

    Forward-fills at most `max_gap_bars` missing silver bars; beyond that the
    value stays NaN (detectors treat NaN as no-event). No-op if the cache file
    is absent or the column already exists.
    """
    if "xag_close" in df.columns:
        return df
    from pathlib import Path
    p = Path(path) if path else (
        Path(__file__).resolve().parents[2] / "data" / "lab_xagusd_cache.csv")
    if not p.exists():
        return df
    xag = pd.read_csv(p, parse_dates=["timestamp"])[["timestamp", "close"]]
    xag = xag.rename(columns={"close": "xag_close"}).sort_values("timestamp")
    out = df.merge(xag, on="timestamp", how="left")
    out["xag_close"] = out["xag_close"].ffill(limit=max_gap_bars)
    return out


def detect_ratio_stretch(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Gold/silver ratio z-score beyond +/- z_thr -> FADE hypothesis on gold
    (ratio rich => short gold, ratio cheap => long gold). Needs `xag_close`
    (see attach_xag); silently fires nothing without it. Ratio uses bar-i
    closes of both legs — knowable at bar-i close."""
    p = _merged(RATIO_STRETCH_DEFAULTS, params)
    out = _frame(len(df))
    if "xag_close" not in df.columns:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = df["close"] / df["xag_close"]
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    mean = ratio.rolling(p["window"], min_periods=p["window"]).mean()
    std = ratio.rolling(p["window"], min_periods=p["window"]).std()
    z = (ratio - mean) / std
    fired = (z.abs() >= p["z_thr"]) & z.notna()
    fired = fired.fillna(False)
    out.loc[fired, "fired"] = True
    out.loc[fired, "direction"] = (-np.sign(z[fired])).astype(np.int8)
    out.loc[fired, "strength"] = z.abs()[fired] - p["z_thr"]
    return out


ASIA_EBREAK_DEFAULTS = {"eu_window_bars_hours": 6}


def detect_asia_range_ebreak(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """First close beyond the completed Asia-session range during the first
    `eu_window_bars_hours` hours of London. Range is frozen before London
    opens (Asia 02:00-10:00 broker time ends at the London 10:00 open), so
    the level is fully causal; one event per day per direction max, first
    break wins."""
    p = _merged(ASIA_EBREAK_DEFAULTS, params)
    out = _frame(len(df))
    ts = df["timestamp"]
    day = ts.dt.normalize()
    asia = session_mask(ts, *SESSIONS["asia"])
    lo_h, _ = SESSIONS["london"]
    eu_end = f"{int(lo_h[:2]) + p['eu_window_bars_hours']:02d}{lo_h[2:]}"
    eu_win = session_mask(ts, lo_h, eu_end)

    asia_hi = df["high"].where(asia).groupby(day).transform("max")
    asia_lo = df["low"].where(asia).groupby(day).transform("min")

    up = eu_win & (df["close"] > asia_hi)
    dn = eu_win & (df["close"] < asia_lo)
    any_evt = (up | dn) & asia_hi.notna() & asia_lo.notna() & df["atr"].notna()
    first = any_evt & (any_evt.groupby(day).cumsum() == 1)

    buy = first & up
    sell = first & dn & ~buy
    out.loc[buy, "fired"] = True
    out.loc[buy, "direction"] = 1
    out.loc[buy, "strength"] = ((df["close"] - asia_hi) / df["atr"])[buy]
    out.loc[sell, "fired"] = True
    out.loc[sell, "direction"] = -1
    out.loc[sell, "strength"] = ((asia_lo - df["close"]) / df["atr"])[sell]
    return out


# ---------------------------------------------------------------- registry

EVENT_REGISTRY = {
    "sweep_high": (detect_sweep_high, SWEEP_DEFAULTS),
    "sweep_low": (detect_sweep_low, SWEEP_DEFAULTS),
    "equal_level_sweep": (detect_equal_level_sweep, EQUAL_SWEEP_DEFAULTS),
    "fvg_bull": (detect_fvg_bull, FVG_DEFAULTS),
    "fvg_bear": (detect_fvg_bear, FVG_DEFAULTS),
    "fvg_fill_bull": (detect_fvg_fill_bull, FVG_FILL_DEFAULTS),
    "fvg_fill_bear": (detect_fvg_fill_bear, FVG_FILL_DEFAULTS),
    "bos_up": (detect_bos_up, BOS_DEFAULTS),
    "bos_down": (detect_bos_down, BOS_DEFAULTS),
    "displacement": (detect_displacement, DISPLACEMENT_DEFAULTS),
    "ob_retest_bull": (detect_ob_retest_bull, OB_RETEST_DEFAULTS),
    "ob_retest_bear": (detect_ob_retest_bear, OB_RETEST_DEFAULTS),
    "range_break": (detect_range_break, RANGE_BREAK_DEFAULTS),
    "judas": (detect_judas, JUDAS_DEFAULTS),
    "session_open_asia": (detect_session_open_asia, SESSION_OPEN_DEFAULTS),
    "session_open_london": (detect_session_open_london, SESSION_OPEN_DEFAULTS),
    "session_open_ny": (detect_session_open_ny, SESSION_OPEN_DEFAULTS),
    "vol_spike": (detect_vol_spike, VOL_SPIKE_DEFAULTS),
    "pdh_touch": (detect_pdh_touch, PD_LEVEL_DEFAULTS),
    "pdl_touch": (detect_pdl_touch, PD_LEVEL_DEFAULTS),
    "pdh_break": (detect_pdh_break, PD_LEVEL_DEFAULTS),
    "pdl_break": (detect_pdl_break, PD_LEVEL_DEFAULTS),
    "vwap_stretch": (detect_vwap_stretch, VWAP_STRETCH_DEFAULTS),
    # 2026-07-10 expansion: retrace/confirmation white space + documented
    # anomaly families (see mining discipline in CLAUDE.md — same gauntlet)
    "orb_break_london": (detect_orb_break_london, ORB_LDN_DEFAULTS),
    "orb_break_ny": (detect_orb_break_ny, ORB_NY_DEFAULTS),
    "orb_pullback_london": (detect_orb_pullback_london, ORB_PB_LDN_DEFAULTS),
    "orb_pullback_ny": (detect_orb_pullback_ny, ORB_PB_NY_DEFAULTS),
    "sweep_reclaim": (detect_sweep_reclaim, SWEEP_RECLAIM_DEFAULTS),
    "failed_break": (detect_failed_break, FAILED_BREAK_DEFAULTS),
    "wick_rejection": (detect_wick_rejection, WICK_REJECT_DEFAULTS),
    "round_level_reject": (detect_round_level_reject, ROUND_LEVEL_DEFAULTS),
    "vol_dryup": (detect_vol_dryup, VOL_DRYUP_DEFAULTS),
    "inside_nr7": (detect_inside_nr7, NR_DEFAULTS),
    "settlement_gap": (detect_settlement_gap, GAP_DEFAULTS),
    "pm_fix_window": (detect_pm_fix_window, PM_FIX_DEFAULTS),
    "news_reopen": (detect_news_reopen, NEWS_REOPEN_DEFAULTS),
    "h1_sweep": (detect_h1_sweep, H1_SWEEP_DEFAULTS),
    # 2026-07-23 expansion: externally-validated geometries (quant-trading-lab
    # analysis), re-implemented and judged under our own gauntlet
    "ratio_stretch": (detect_ratio_stretch, RATIO_STRETCH_DEFAULTS),
    "asia_range_ebreak": (detect_asia_range_ebreak, ASIA_EBREAK_DEFAULTS),
}


def run_all(df: pd.DataFrame, names=None, overrides: dict = None) -> dict:
    """Run detectors by name (default: all). overrides = {name: params}."""
    names = names or list(EVENT_REGISTRY)
    overrides = overrides or {}
    out = {}
    for name in names:
        fn, _defaults = EVENT_REGISTRY[name]
        out[name] = fn(df, overrides.get(name))
    return out
