#!/usr/bin/env python3
"""
Database migration script to update User table schema
"""
import asyncio
import sqlite3
import os
import sys

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import AsyncSessionLocal, engine
from app.models import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def migrate_database():
    """Migrate database to new schema"""
    try:
        # For SQLite, we'll recreate the tables with the new schema
        logger.info("Starting database migration...")
        
        # Create all tables with new schema
        async with engine.begin() as conn:
            # Drop existing tables and recreate (for development)
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
            
        logger.info("Database migration completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(migrate_database()) 