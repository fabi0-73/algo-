"""
Database Setup Script
Automatically creates the database and tables for AMD backtester.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from psycopg2 import sql
from psycopg2.errors import DuplicateDatabase
from dotenv import load_dotenv

load_dotenv()


def check_postgresql_installed():
    """Check if PostgreSQL is accessible."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            database="postgres"
        )
        conn.close()
        return True
    except psycopg2.OperationalError as e:
        print(f"Cannot connect to PostgreSQL: {e}")
        print("\nPlease make sure:")
        print("  1. PostgreSQL is installed")
        print("  2. PostgreSQL service is running")
        print("  3. Your .env file has correct credentials")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def create_database():
    """Create the amd_trading database if it doesn't exist."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            database="postgres"
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        db_name = os.getenv("DB_NAME", "amd_trading")
        
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_name,)
        )
        exists = cursor.fetchone()
        
        if exists:
            print(f"Database '{db_name}' already exists")
        else:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
            print(f"Database '{db_name}' created successfully!")
        
        cursor.close()
        conn.close()
        return True
        
    except DuplicateDatabase:
        print(f"Database already exists")
        return True
    except Exception as e:
        print(f"Error creating database: {e}")
        return False


def create_tables():
    """Create all required tables."""
    try:
        from src.data.db import Database
        
        db = Database()
        db.create_tables()
        print("Tables created successfully!")
        return True
    except Exception as e:
        print(f"Error creating tables: {e}")
        return False


def test_connection():
    """Test the database connection."""
    try:
        from src.data.db import Database
        
        db = Database()
        session = db.get_session()
        session.close()
        print("Database connection test passed!")
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("AMD Strategy - Database Setup")
    print("=" * 60)
    print()
    
    print("Step 1: Checking PostgreSQL installation...")
    if not check_postgresql_installed():
        print("\nSetup failed. Please check PostgreSQL.")
        return False
    print("PostgreSQL is accessible")
    print()
    
    print("Step 2: Creating database...")
    if not create_database():
        return False
    print()
    
    print("Step 3: Creating tables...")
    if not create_tables():
        return False
    print()
    
    print("Step 4: Testing connection...")
    if not test_connection():
        return False
    print()
    
    print("=" * 60)
    print("Database setup complete!")
    print("=" * 60)
    print("\nYou can now run:")
    print("  python scripts/fetch_data.py --months 6")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
