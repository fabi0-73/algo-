"""
Generate a best-effort HIGH-impact USD news calendar for the news filter.

Timestamps are written in the BROKER time frame the candle data uses (IC Markets
server = New York + 7h; see the TIME_CONFIG note in config.py). Because the broker
tracks US DST, a US event at a fixed New-York wall-clock time maps to a FIXED
broker-clock time regardless of DST:
    NFP / CPI   08:30 ET  ->  15:30 (broker)
    FOMC        14:00 ET  ->  21:00 (broker)

Covers the reliably-known scheduled movers:
  - NFP: 1st Friday of each month (computed)
  - FOMC rate decisions: scheduled decision days (hardcoded, verify vs Fed calendar)
  - CPI: monthly release days (hardcoded best-effort, verify vs BLS calendar)

This is NOT a complete calendar. For production accuracy, augment data/news_events.csv
with a real economic-calendar export (ForexFactory CSV / MT5 calendar): PPI, Retail
Sales, PCE, FOMC minutes, Fed Chair testimony, etc. A wrong date only mis-places a
~20-minute blackout window, so this is safe to iterate on.

Usage:
  python scripts/generate_news_events.py                # data range, default output
  python scripts/generate_news_events.py --start 2024-09-01 --end 2026-03-01
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
from datetime import datetime, timedelta

from config import NEWS_FILTER

BROKER_OFFSET_FROM_ET = 7  # IC Markets server = New York + 7h (constant; broker tracks US DST)

# FOMC rate-decision days (decision released ~14:00 ET). Best-effort — verify vs Fed calendar.
FOMC_DATES = [
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-16",
]

# CPI release days (~08:30 ET). Best-effort — verify vs BLS calendar.
# A date off by a day only mis-places a short blackout window; a MISSING
# month leaves the filter blind (which is how the scanner ran unprotected
# Feb->Jul 2026) — so keep these lists extended past the data horizon.
CPI_DATES = [
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13",
    "2025-06-11", "2025-07-15", "2025-08-12", "2025-09-11", "2025-10-15",
    "2025-11-13", "2025-12-10",
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-10", "2026-05-12",
    "2026-06-10", "2026-07-14", "2026-08-12", "2026-09-11", "2026-10-13",
    "2026-11-12", "2026-12-10",
    "2027-01-13", "2027-02-10", "2027-03-10", "2027-04-13", "2027-05-12",
    "2027-06-10",
]


def first_friday(year: int, month: int) -> datetime:
    d = datetime(year, month, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7)  # weekday(): Mon=0..Fri=4


def month_range(start: datetime, end: datetime):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def to_broker(day: datetime, et_hour: int, et_min: int) -> datetime:
    """ET wall-clock time on `day` shifted into the broker frame (+7h)."""
    base = day.replace(hour=et_hour, minute=et_min, second=0, microsecond=0)
    return base + timedelta(hours=BROKER_OFFSET_FROM_ET)


def build(start: datetime, end: datetime):
    rows = []
    for y, m in month_range(start, end):
        rows.append((to_broker(first_friday(y, m), 8, 30), "USD", "HIGH", "NFP (Non-Farm Payrolls)"))
    for ds in FOMC_DATES:
        rows.append((to_broker(datetime.strptime(ds, "%Y-%m-%d"), 14, 0), "USD", "HIGH", "FOMC Rate Decision"))
    for ds in CPI_DATES:
        rows.append((to_broker(datetime.strptime(ds, "%Y-%m-%d"), 8, 30), "USD", "HIGH", "CPI (Consumer Price Index)"))
    rows = [r for r in rows if start <= r[0] <= end]
    rows.sort(key=lambda r: r[0])
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate best-effort HIGH-impact USD news CSV (broker time frame)")
    parser.add_argument("--start", type=str, default="2024-09-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2026-03-01", help="End date YYYY-MM-DD")
    parser.add_argument("--out", type=str, default=NEWS_FILTER["csv_path"], help="Output CSV path")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    rows = build(start, end)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "currency", "impact", "title"])
        for ts, ccy, impact, title in rows:
            w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), ccy, impact, title])

    print(f"Wrote {len(rows)} events to {args.out}")
    print(f"Range: {rows[0][0]} -> {rows[-1][0]} (broker frame, NY+7)")
    by_title = {}
    for _, _, _, t in rows:
        key = t.split(" (")[0]
        by_title[key] = by_title.get(key, 0) + 1
    for k, v in by_title.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
