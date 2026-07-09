"""
Data-quality audit for the M5 cache — the gate before any research run.

Works on the cache frame alone (timestamp/open/high/low/close/volume; no
bid/ask is stored, so spread sanity is out of scope and stated as such in the
report). Timestamps are broker time, open-time-stamped bars.

Gap taxonomy (broker ~NY+7, daily settlement break at hour 0):
  weekend    — gap that spans a Saturday (Fri close -> Mon/Sun reopen)
  settlement — intraday gap that crosses broker midnight, <= max_settle_hours
  anomalous  — everything else (missing data, broker outage)
"""
import numpy as np
import pandas as pd

from .strategies.base import session_mask

FREQ_MINUTES = 5
MAX_SETTLE_HOURS = 3.0
OUTLIER_SIGMA = 8.0

SESSIONS = {"asia": ("02:00", "10:00"), "london": ("10:00", "16:30"),
            "ny": ("16:30", "23:59")}


def audit_m5(df: pd.DataFrame) -> dict:
    """Full audit dict; every check is cache-only. Keys are stable so
    audit.json diffs across fetches are meaningful."""
    ts = pd.to_datetime(df["timestamp"])
    n = len(df)
    out = {"n_bars": int(n)}
    if n == 0:
        return out
    out["start"] = str(ts.iloc[0])
    out["end"] = str(ts.iloc[-1])

    out["duplicate_timestamps"] = int(ts.duplicated().sum())
    out["non_monotonic"] = int((ts.diff().dt.total_seconds() < 0).sum())
    out["off_grid"] = int(((ts.dt.minute % FREQ_MINUTES != 0)
                           | (ts.dt.second != 0)).sum())

    bad_hl = df["high"] < df["low"]
    bad_hi = df["high"] < df[["open", "close"]].max(axis=1)
    bad_lo = df["low"] > df[["open", "close"]].min(axis=1)
    out["ohlc_violations"] = int((bad_hl | bad_hi | bad_lo).sum())
    out["zero_range_bars"] = int((df["high"] == df["low"]).sum())
    out["zero_volume_bars"] = int((df["volume"] == 0).sum())

    ret = np.log(df["close"] / df["close"].shift(1))
    sigma = ret.std()
    outliers = df.loc[ret.abs() > OUTLIER_SIGMA * sigma, "timestamp"]
    out["return_outliers"] = {
        "sigma_threshold": OUTLIER_SIGMA,
        "count": int(len(outliers)),
        "top": [str(t) for t in outliers.head(10)],
    }

    # ---- gap census
    delta_min = ts.diff().dt.total_seconds() / 60.0
    gaps = pd.DataFrame({
        "prev": ts.shift(1), "cur": ts, "minutes": delta_min,
    }).iloc[1:]
    gaps = gaps[gaps["minutes"] > FREQ_MINUTES]
    weekend = gaps.apply(_spans_saturday, axis=1) if len(gaps) else pd.Series(dtype=bool)
    crosses_midnight = (gaps["prev"].dt.normalize() != gaps["cur"].dt.normalize()) \
        & (gaps["minutes"] <= MAX_SETTLE_HOURS * 60)
    settlement = ~weekend & crosses_midnight if len(gaps) else pd.Series(dtype=bool)
    anomalous = gaps[~weekend & ~settlement] if len(gaps) else gaps
    out["gaps"] = {
        "total": int(len(gaps)),
        "weekend": int(weekend.sum()) if len(gaps) else 0,
        "settlement": int(settlement.sum()) if len(gaps) else 0,
        "anomalous": int(len(anomalous)),
        "top_anomalous": [
            {"from": str(r["prev"]), "to": str(r["cur"]),
             "minutes": float(r["minutes"])}
            for _, r in anomalous.nlargest(20, "minutes").iterrows()
        ],
    }

    # ---- coverage
    day = ts.dt.normalize()
    per_day = day.value_counts()
    out["bars_per_day"] = {
        "mean": float(per_day.mean()), "min": int(per_day.min()),
        "max": int(per_day.max()), "n_days": int(len(per_day)),
        "thin_days": int((per_day < 200).sum()),  # full weekday ~276-288 bars
    }
    out["bars_per_session"] = {
        name: float(session_mask(ts, s, e).groupby(day).sum().mean())
        for name, (s, e) in SESSIONS.items()
    }
    month = ts.dt.strftime("%Y-%m")
    out["monthly_coverage"] = month.value_counts().sort_index().astype(int).to_dict()

    out["notes"] = ["spread sanity not checkable: cache stores no bid/ask"]
    return out


def _spans_saturday(row) -> bool:
    """True if any calendar day strictly inside (prev, cur] is a Saturday,
    or the gap starts Friday and ends after Friday."""
    days = pd.date_range(row["prev"].normalize(), row["cur"].normalize(), freq="D")
    return any(d.dayofweek == 5 for d in days)


def format_report(audit: dict) -> str:
    """Human-readable console summary of audit_m5() output."""
    lines = ["M5 CACHE AUDIT", "=" * 40]
    if audit.get("n_bars", 0) == 0:
        return "\n".join(lines + ["EMPTY FRAME"])
    lines.append(f"bars {audit['n_bars']}  {audit['start']} .. {audit['end']}")
    lines.append(f"dupes {audit['duplicate_timestamps']}  "
                 f"non-monotonic {audit['non_monotonic']}  "
                 f"off-grid {audit['off_grid']}")
    lines.append(f"OHLC violations {audit['ohlc_violations']}  "
                 f"zero-range {audit['zero_range_bars']}  "
                 f"zero-volume {audit['zero_volume_bars']}")
    ro = audit["return_outliers"]
    lines.append(f"return outliers (> {ro['sigma_threshold']} sigma): {ro['count']}")
    g = audit["gaps"]
    lines.append(f"gaps: {g['total']} total = {g['weekend']} weekend + "
                 f"{g['settlement']} settlement + {g['anomalous']} ANOMALOUS")
    for item in g["top_anomalous"][:5]:
        lines.append(f"  anomalous {item['from']} -> {item['to']} "
                     f"({item['minutes']:.0f} min)")
    b = audit["bars_per_day"]
    lines.append(f"bars/day mean {b['mean']:.1f} min {b['min']} max {b['max']} "
                 f"days {b['n_days']} thin(<200) {b['thin_days']}")
    months = audit["monthly_coverage"]
    lines.append(f"months covered: {len(months)} "
                 f"({min(months)} .. {max(months)})")
    for note in audit["notes"]:
        lines.append(f"note: {note}")
    return "\n".join(lines)
