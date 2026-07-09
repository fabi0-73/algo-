"""
Data-quality audit of the M5 research cache — run before any research run;
anomalous gaps or corrupt bars invalidate everything downstream.

Usage:
    python scripts/audit_data.py                          # default cache / DB
    python scripts/audit_data.py --cache data/lab_m5_cache.csv.gz
"""
import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.audit import audit_m5, format_report
from src.research.mtf import load_m5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="explicit cache file (.csv/.csv.gz)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="reports/research")
    ap.add_argument("--no-json", action="store_true")
    args = ap.parse_args()

    df = load_m5(args.start, args.end, cache_path=args.cache)
    audit = audit_m5(df)
    print(format_report(audit))

    if not args.no_json:
        run_id = uuid.uuid4().hex[:8]
        out_dir = Path(args.out) / f"data_audit_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "audit.json", "w") as f:
            json.dump(audit, f, indent=2)
        print(f"\naudit: {out_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
