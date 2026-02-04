"""
Quick Database Connection Test
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.data.db import Database


def main():
    print("Testing database connection...")
    try:
        db = Database()
        db.create_tables()
        
        count = db.get_candle_count("XAUUSD", "M5")
        print(f"Connection successful!")
        print(f"Current candles in DB: {count}")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL is installed and running")
        print("  2. Your .env file has correct credentials")
        print("  3. Database 'amd_trading' exists")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
