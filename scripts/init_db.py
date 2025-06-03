#!/usr/bin/env python3
"""
Database initialization script
"""
import asyncio
import sys
import os

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import engine, Base
from app.models import User, Chat, Message
from app.config import settings

async def init_database():
    """Initialize the database with tables"""
    try:
        print("Initializing database...")
        print(f"Database URL: {settings.database_url}")
        
        async with engine.begin() as conn:
            # Drop all tables (uncomment if you want to reset)
            # await conn.run_sync(Base.metadata.drop_all)
            # print("Dropped existing tables")
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            print("Created database tables successfully")
            
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False
    
    return True

async def create_sample_user():
    """Create a sample user for testing"""
    from app.database import AsyncSessionLocal
    
    try:
        async with AsyncSessionLocal() as session:
            # Check if user already exists
            from sqlalchemy import select
            result = await session.execute(select(User).where(User.username == "testuser"))
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                print(f"User 'testuser' already exists with ID: {existing_user.id}")
                return existing_user.id
            
            # Create new user
            user = User(username="testuser")
            session.add(user)
            await session.commit()
            await session.refresh(user)
            
            print(f"Created sample user 'testuser' with ID: {user.id}")
            return user.id
            
    except Exception as e:
        print(f"Error creating sample user: {e}")
        return None

async def main():
    """Main function"""
    print("=== Database Initialization ===")
    
    # Initialize database
    success = await init_database()
    if not success:
        print("Failed to initialize database")
        return
    
    # Create sample user
    user_id = await create_sample_user()
    if user_id:
        print(f"Sample user created successfully. User ID: {user_id}")
    
    print("=== Initialization Complete ===")

if __name__ == "__main__":
    asyncio.run(main()) 