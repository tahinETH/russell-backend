from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError
from app.database import AsyncSessionLocal
from app.models import User
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class UserDataRepository:
    """Repository for user data operations"""
    
    async def create_user(
        self,
        user_id: str,
        email: str,
        name: Optional[str] = None,
        username: Optional[str] = None,
        
        fe_metadata: Optional[Dict[str, Any]] = None
    ) -> User:
        """Create a new user"""
        async with AsyncSessionLocal() as session:
            try:
                user = User(
                    id=user_id,
                    email=email,
                    name=name,
                    username=username,
                    
                    fe_metadata=fe_metadata
                )
                
                session.add(user)
                await session.commit()
                await session.refresh(user)
                
                logger.info(f"Created user {user_id} with email {email}")
                return user
                
            except IntegrityError as e:
                await session.rollback()
                logger.error(f"Failed to create user {user_id}: {str(e)}")
                raise
            except Exception as e:
                await session.rollback()
                logger.error(f"Unexpected error creating user {user_id}: {str(e)}")
                raise

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(User).where(User.id == user_id)
                )
                user = result.scalar_one_or_none()
                return user
            except Exception as e:
                logger.error(f"Error getting user {user_id}: {str(e)}")
                raise

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(User).where(User.email == email)
                )
                user = result.scalar_one_or_none()
                return user
            except Exception as e:
                logger.error(f"Error getting user by email {email}: {str(e)}")
                raise

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(User).where(User.username == username)
                )
                user = result.scalar_one_or_none()
                return user
            except Exception as e:
                logger.error(f"Error getting user by username {username}: {str(e)}")
                raise

    async def update_user(
        self,
        user_id: str,
        **updates
    ) -> Optional[User]:
        """Update user with given fields"""
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(User).where(User.id == user_id)
                )
                user = result.scalar_one_or_none()
                
                if not user:
                    logger.warning(f"User {user_id} not found for update")
                    return None
                
                # Update only provided fields
                for field, value in updates.items():
                    if hasattr(user, field):
                        setattr(user, field, value)
                
                await session.commit()
                await session.refresh(user)
                
                logger.info(f"Updated user {user_id}")
                return user
                
            except IntegrityError as e:
                await session.rollback()
                logger.error(f"Failed to update user {user_id}: {str(e)}")
                raise
            except Exception as e:
                await session.rollback()
                logger.error(f"Unexpected error updating user {user_id}: {str(e)}")
                raise

    async def delete_user(self, user_id: str) -> bool:
        """Delete user by ID"""
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(User).where(User.id == user_id)
                )
                user = result.scalar_one_or_none()
                
                if not user:
                    logger.warning(f"User {user_id} not found for deletion")
                    return False
                
                await session.delete(user)
                await session.commit()
                
                logger.info(f"Deleted user {user_id}")
                return True
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error deleting user {user_id}: {str(e)}")
                raise

    async def user_exists(self, user_id: str) -> bool:
        """Check if user exists"""
        user = await self.get_user(user_id)
        return user is not None 