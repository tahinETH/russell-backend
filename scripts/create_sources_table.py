#!/usr/bin/env python3
"""
Script to create the sources table in the database
"""
import sys
import os
import asyncio

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import engine, Base
from app.models import Source

async def main():
    """Create the sources table"""
    print("Creating sources table...")
    
    try:
        # Create all tables (this will only create new ones)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        print("Sources table created successfully!")
        return True
        
    except Exception as e:
        print(f"Error creating sources table: {e}")
        return False

if __name__ == "__main__":
    # Run the main function
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 