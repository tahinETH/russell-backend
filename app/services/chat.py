from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, Dict, Any
import uuid
import logging

from ..models import User, Chat, Message
from ..database import AsyncSessionLocal

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        pass
    
    async def verify_user_exists(self, user_id: str) -> bool:
        """Verify if user exists in database"""
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()
            return user is not None
    
    async def get_or_create_chat(self, user_id: str, chat_id: Optional[str] = None) -> Optional[Chat]:
        """Get existing chat or create a new one"""
        async with AsyncSessionLocal() as db:
            if chat_id:
                try:
                    # Convert string UUID to UUID object
                    chat_uuid = uuid.UUID(chat_id)
                    result = await db.execute(
                        select(Chat).where(
                            Chat.id == chat_uuid,
                            Chat.user_id == user_id
                        )
                    )
                    chat = result.scalar_one_or_none()
                    return chat
                except ValueError:
                    # Invalid UUID format
                    logger.error(f"Invalid chat_id format: {chat_id}")
                    return None
            else:
                chat = Chat(user_id=user_id)
                db.add(chat)
                await db.commit()
                await db.refresh(chat)
                return chat
    
    async def save_user_message(self, chat_id: uuid.UUID, content: str) -> Message:
        """Save a user message to the database"""
        async with AsyncSessionLocal() as db:
            user_message = Message(
                chat_id=chat_id,
                role="user",
                content=content
            )
            db.add(user_message)
            await db.commit()
            await db.refresh(user_message)
            return user_message
    
    async def save_assistant_message(
        self, 
        chat_id: uuid.UUID, 
        content: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Save an assistant message to the database"""
        async with AsyncSessionLocal() as db:
            assistant_message = Message(
                chat_id=chat_id,
                role="assistant",
                content=content,
                context=context or {}
            )
            db.add(assistant_message)
            await db.commit()
            await db.refresh(assistant_message)
            return assistant_message
    
    async def update_chat_name(self, chat_id: uuid.UUID, name: str) -> bool:
        """Update the name of a chat"""
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(select(Chat).where(Chat.id == chat_id))
                chat = result.scalar_one_or_none()
                if chat:
                    chat.name = name
                    await db.commit()
                    logger.info(f"Updated chat {chat_id} name to: {name}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to update chat name: {e}")
            return False
    
    async def is_new_chat(self, chat_id: uuid.UUID) -> bool:
        """Check if this is a new chat (has only one message - the user message)"""
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(Message).where(Message.chat_id == chat_id)
                )
                messages = result.scalars().all()
                return len(messages) <= 1  # Only user message exists
        except Exception as e:
            logger.error(f"Failed to check if chat is new: {e}")
            return False 