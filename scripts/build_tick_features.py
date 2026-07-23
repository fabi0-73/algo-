"""Tick microstructure pipeline (quote-only feed):

1. Aggregate monthly tick parquets into per-M5-bar features
   -> data/tick_features_m5.parquet
2. Own-feed cost calibration: spread percentiles + quote-gap distribution
   (the numbers that replace the borrowed stress-gauntlet tiers).
3. PRE-REGISTERED conditioning test on the adopted config's trades
   (report a1428430), hypotheses fixed before looking:
     H1: louder manipulation sweeps (tick-burst percentile) -> better trades
     H2: wider spread at entry -> worse trades
   n is small (~last 12 months of trades) — indicative only, no promotion.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TICKS = sorted((ROOT / "data" / "ticks").glob("XAUUSD_*.parquet"))
OUT = ROOT / "data" / "tick_features_m5.parquet"


def month_features(path: Path) -> pd.DataFrame:
    t = pd.read_parquet(path)
    ts = pd.to_datetime(t["time_msc"], unit="ms")
    mid = (t["bid"].to_numpy() + t["ask"].to_numpy()) / 2
    spread = (t["ask"] - t["bid"]).to_numpy()
    gap_ms = np.diff(t["time_msc"].to_numpy(), prepend=t["time_msc"].iloc[0])
    tick_dir = np.sign(np.diff(mid, prepend=mid[0]))

    df = pd.DataFrame({
        "bar": ts.dt.floor("5min"),
        "sec10": ts.dt.floor("10s"),
        "spread": spread,
        "gap_ms": gap_ms,
        "up": tick_dir > 0,
        "dn": tick_dir < 0,
    })
    burst = (df.groupby(["bar", "sec10"]).size()
               .groupby("bar").max().rename("burst_10s"))
    g = df.groupby("bar")
    out = pd.DataFrame({
        "n_ticks": g.size(),
        "spread_mean": g["spread"].mean(),
        "spread_p95": g["spread"].quantile(0.95),
        "spread_max": g["spread"].max(),
        "gap_max_ms": g["gap_ms"].max(),
        "imbalance": (g["up"].sum() - g["dn"].sum()) / g.size(),
    }).join(burst)
    return out.reset_index().rename(columns={"bar": "timestamp"})


def main():
    frames = []
    for p in TICKS:
        f = month_features(p)
        frames.append(f)
        print(f"{p.name}: {len(f)} bars")
    feats = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    feats = feats.drop_duplicates(subset="timestamp", keep="first")
    feats.to_parquet(OUT, index=False)
    print(f"\nfeatures: {len(feats)} M5 bars -> {OUT.name}")

    # ---- 2. own-feed cost calibration ------------------------------------
    print("\n=== OWN-FEED COST CALIBRATION (per 0.01 lot = 1 oz) ===")
    sp = feats["spread_mean"]
    for q, lbl in ((0.75, "base"), (0.95, "adverse"), (0.99, "severe")):
        pts = sp.quantile(q) * 100
        print(f"  spread {lbl:8s} (p{int(q*100)}): {pts:5.1f} points "
              f"= ${sp.quantile(q):.3f}/oz")
    big_gaps = feats.loc[feats["gap_max_ms"] >= 30_000, "gap_max_ms"]
    print(f"  bars with >=30s quote gap: {len(big_gaps)} "
          f"({100*len(big_gaps)/len(feats):.2f}% of bars)")

    # ---- 3. pre-registered conditioning test -----------------------------
    print("\n=== AMD TRADE CONDITIONING (adopted config a1428430) ===")
    j = json.loads((ROOT / "reports" / "backtest_a1428430" /
                    "results.json").read_text())
    trades = pd.DataFrame(j["trades"])
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    t0 = feats["timestamp"].min()
    tt = trades[trades["entry_time"] >= t0].copy()
    print(f"trades with tick coverage: {len(tt)} of {len(trades)}")

    feats_idx = feats.set_index("timestamp")
    # entry-bar features
    entry_bar = tt["entry_time"].dt.floor("5min")
    for col in ("burst_10s", "spread_mean", "n_ticks"):
        tt[f"entry_{col}"] = entry_bar.map(feats_idx[col]).to_numpy()
    # sweep loudness: max burst over the 6 bars ending at entry (the
    # manipulation->retest leg approximation available without per-trade
    # sweep timestamps)
    burst_roll = (feats_idx["burst_10s"].rolling("30min").max())
    tt["sweep_burst"] = entry_bar.map(burst_roll).to_numpy()

    def slice_report(name, series, r):
        ok = series.notna()
        s, rr = series[ok], r[ok]
        if len(s) < 20:
            print(f"  {name}: insufficient joined trades ({len(s)})")
            return
        terc = pd.qcut(s, 3, labels=["low", "mid", "high"], duplicates="drop")
        print(f"  {name} (n={len(s)}):")
        for lab in terc.cat.categories:
            m = terc == lab
            print(f"    {lab:4s}: n={m.sum():3d}  expR {rr[m].mean():+.3f}  "
                  f"WR {(rr[m] > 0).mean()*100:.0f}%")

    r = tt["r_multiple"]
    print("\nH1 — sweep loudness (30-min max tick burst before entry):")
    slice_report("sweep_burst", tt["sweep_burst"], r)
    print("\nH2 — spread at entry bar:")
    slice_report("entry_spread", tt["entry_spread_mean"], r)


if __name__ == "__main__":
    main()
