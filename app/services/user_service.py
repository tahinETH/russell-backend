from typing import Optional
import uuid
import logging
from db.users import UserDataRepository
from ..models import User, UserCreate, UserResponse

logger = logging.getLogger(__name__)

class UserService:
    """Service layer for user-related operations"""
    
    def __init__(self):
        self.user_repo = UserDataRepository()
    
    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user with validation"""
        try:
            # Check if username already exists (if provided)
            if user_data.username:
                existing_user = await self.user_repo.get_user_by_username(user_data.username)
                if existing_user:
                    raise ValueError("Username already exists")
            
            # Check if email already exists
            existing_user = await self.user_repo.get_user_by_email(user_data.email)
            if existing_user:
                raise ValueError("Email already exists")
            
            # Generate UUID for user
            user_id = str(uuid.uuid4())
            
            # Create user
            user = await self.user_repo.create_user(
                user_id=user_id,
                email=user_data.email,
                name=user_data.name,
                username=user_data.username,
                fe_metadata=user_data.fe_metadata
            )
            
            return UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                name=user.name,
                fe_metadata=user.fe_metadata,
                created_at=user.created_at
            )
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise Exception("Failed to create user")
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            return await self.user_repo.get_user(user_id)
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {e}")
            raise
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            return await self.user_repo.get_user_by_username(username)
        except Exception as e:
            logger.error(f"Error getting user by username {username}: {e}")
            raise
    
    async def user_exists(self, user_id: str) -> bool:
        """Check if user exists"""
        try:
            user = await self.get_user(user_id)
            return user is not None
        except Exception as e:
            logger.error(f"Error checking if user exists {user_id}: {e}")
            return False 