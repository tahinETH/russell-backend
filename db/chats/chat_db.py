from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import IntegrityError
from app.database import AsyncSessionLocal
from app.models import Chat, Message, MessageImage, User
import logging
from typing import Optional, List, Dict, Any
import uuid

logger = logging.getLogger(__name__)

class ChatDataRepository:
    """Repository for chat and message data operations"""
    
    async def create_chat(self, user_id: str) -> Chat:
        """Create a new chat for a user"""
        async with AsyncSessionLocal() as session:
            try:
                chat = Chat(user_id=user_id)
                session.add(chat)
                await session.commit()
                await session.refresh(chat)
                
                logger.info(f"Created chat {chat.id} for user {user_id}")
                return chat
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error creating chat for user {user_id}: {str(e)}")
                raise

    async def get_chat(self, chat_id: uuid.UUID, user_id: Optional[str] = None) -> Optional[Chat]:
        """Get chat by ID, optionally filtered by user_id"""
        async with AsyncSessionLocal() as session:
            try:
                query = select(Chat).where(Chat.id == chat_id)
                if user_id:
                    query = query.where(Chat.user_id == user_id)
                
                result = await session.execute(query)
                chat = result.scalar_one_or_none()
                return chat
            except Exception as e:
                logger.error(f"Error getting chat {chat_id}: {str(e)}")
                raise

    async def get_chat_with_messages(self, chat_id: uuid.UUID, user_id: Optional[str] = None) -> Optional[Chat]:
        """Get chat with all its messages and images"""
        async with AsyncSessionLocal() as session:
            try:
                query = select(Chat).options(
                    selectinload(Chat.messages).selectinload(Message.images)
                ).where(Chat.id == chat_id)
                if user_id:
                    query = query.where(Chat.user_id == user_id)
                
                result = await session.execute(query)
                chat = result.scalar_one_or_none()
                return chat
            except Exception as e:
                logger.error(f"Error getting chat {chat_id} with messages: {str(e)}")
                raise

    async def get_user_chats(self, user_id: str) -> List[Chat]:
        """Get all chats for a user"""
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(Chat)
                    .where(Chat.user_id == user_id)
                    .order_by(Chat.created_at.desc())
                )
                chats = result.scalars().all()
                return list(chats)
            except Exception as e:
                logger.error(f"Error getting chats for user {user_id}: {str(e)}")
                raise

    async def create_message(
        self,
        chat_id: uuid.UUID,
        role: str,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Create a new message in a chat"""
        async with AsyncSessionLocal() as session:
            try:
                message = Message(
                    chat_id=chat_id,
                    role=role,
                    content=content,
                    context=context
                )
                session.add(message)
                await session.commit()
                await session.refresh(message)
                
                logger.info(f"Created message {message.id} in chat {chat_id}")
                return message
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error creating message in chat {chat_id}: {str(e)}")
                raise

    async def create_message_image(
        self,
        message_id: uuid.UUID,
        prompt: str,
        image_url: str
    ) -> MessageImage:
        """Create a new image associated with a message"""
        async with AsyncSessionLocal() as session:
            try:
                message_image = MessageImage(
                    message_id=message_id,
                    prompt=prompt,
                    image_url=image_url
                )
                session.add(message_image)
                await session.commit()
                await session.refresh(message_image)
                
                logger.info(f"Created image {message_image.id} for message {message_id}")
                return message_image
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error creating image for message {message_id}: {str(e)}")
                raise

    async def get_chat_messages(self, chat_id: uuid.UUID) -> List[Message]:
        """Get all messages for a chat with their images"""
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(Message)
                    .options(selectinload(Message.images))
                    .where(Message.chat_id == chat_id)
                    .order_by(Message.created_at.asc())
                )
                messages = result.scalars().all()
                return list(messages)
            except Exception as e:
                logger.error(f"Error getting messages for chat {chat_id}: {str(e)}")
                raise

    async def get_message_images(self, message_id: uuid.UUID) -> List[MessageImage]:
        """Get all images for a specific message"""
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(MessageImage)
                    .where(MessageImage.message_id == message_id)
                    .order_by(MessageImage.created_at.asc())
                )
                images = result.scalars().all()
                return list(images)
            except Exception as e:
                logger.error(f"Error getting images for message {message_id}: {str(e)}")
                raise

    async def update_chat_name(self, chat_id: uuid.UUID, name: str) -> bool:
        """Update chat name"""
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(Chat).where(Chat.id == chat_id)
                )
                chat = result.scalar_one_or_none()
                
                if not chat:
                    logger.warning(f"Chat {chat_id} not found for name update")
                    return False
                
                chat.name = name
                await session.commit()
                
                logger.info(f"Updated chat {chat_id} name to: {name}")
                return True
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error updating chat {chat_id} name: {str(e)}")
                raise

    async def delete_chat(self, chat_id: uuid.UUID) -> bool:
        """Delete a chat and all its messages and images"""
        async with AsyncSessionLocal() as session:
            try:
                # First get all messages for this chat
                messages_result = await session.execute(
                    select(Message).where(Message.chat_id == chat_id)
                )
                messages = messages_result.scalars().all()
                
                # Delete all images for these messages
                for message in messages:
                    await session.execute(
                        select(MessageImage).where(MessageImage.message_id == message.id)
                    )
                
                # Delete all messages (this will cascade to images due to foreign key)
                await session.execute(
                    select(Message).where(Message.chat_id == chat_id)
                )
                
                # Then delete the chat
                result = await session.execute(
                    select(Chat).where(Chat.id == chat_id)
                )
                chat = result.scalar_one_or_none()
                
                if not chat:
                    logger.warning(f"Chat {chat_id} not found for deletion")
                    return False
                
                await session.delete(chat)
                await session.commit()
                
                logger.info(f"Deleted chat {chat_id}")
                return True
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error deleting chat {chat_id}: {str(e)}")
                raise

    async def chat_exists(self, chat_id: uuid.UUID, user_id: Optional[str] = None) -> bool:
        """Check if chat exists"""
        chat = await self.get_chat(chat_id, user_id)
        return chat is not None 