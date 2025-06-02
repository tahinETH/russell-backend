#!/usr/bin/env python3
"""
Migration script to add custom_system_prompt column to users table
and populate existing users with default prompt
"""
import asyncio
import sys
import os

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import AsyncSessionLocal, engine
from app.models import User
from app.services.prompts import prepare_query_system_prompt
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def add_custom_prompt_column():
    """Add custom_system_prompt column to users table if it doesn't exist"""
    try:
        async with engine.begin() as conn:
            # Check if column exists
            result = await conn.execute(text("""
                SELECT COUNT(*) 
                FROM pragma_table_info('users') 
                WHERE name = 'custom_system_prompt'
            """))
            column_exists = result.scalar() > 0
            
            if not column_exists:
                logger.info("Adding custom_system_prompt column to users table...")
                await conn.execute(text("""
                    ALTER TABLE users 
                    ADD COLUMN custom_system_prompt TEXT
                """))
                logger.info("Successfully added custom_system_prompt column")
            else:
                logger.info("custom_system_prompt column already exists")
                
    except Exception as e:
        logger.error(f"Error adding column: {e}")
        raise

async def populate_default_prompts():
    """Populate existing users with default custom system prompt"""
    try:
        default_prompt = prepare_query_system_prompt()
        
        async with AsyncSessionLocal() as session:
            # Get all users with null custom_system_prompt
            result = await session.execute(text("""
                SELECT id FROM users 
                WHERE custom_system_prompt IS NULL
            """))
            user_ids = [row[0] for row in result.fetchall()]
            
            if user_ids:
                logger.info(f"Populating default prompts for {len(user_ids)} users...")
                
                # Update users with default prompt
                await session.execute(text("""
                    UPDATE users 
                    SET custom_system_prompt = :default_prompt 
                    WHERE custom_system_prompt IS NULL
                """), {"default_prompt": default_prompt})
                
                await session.commit()
                logger.info(f"Successfully populated default prompts for {len(user_ids)} users")
            else:
                logger.info("No users need default prompt population")
                
    except Exception as e:
        logger.error(f"Error populating default prompts: {e}")
        raise

async def main():
    """Main migration function"""
    logger.info("=== Custom System Prompt Migration ===")
    
    try:
        # Step 1: Add column if it doesn't exist
        await add_custom_prompt_column()
        
        # Step 2: Populate existing users with default prompt
        await populate_default_prompts()
        
        logger.info("=== Migration completed successfully ===")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 