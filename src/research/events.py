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
    prev_in = in_sess.shift(1).fillna(False) & (day == day.shift(1))
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
    first = above & ~above.groupby(day).cummax().shift(1).where(day == day.shift(1), False).fillna(False)
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
    first = below & ~below.groupby(day).cummax().shift(1).where(day == day.shift(1), False).fillna(False)
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
