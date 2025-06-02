from typing import Optional, List, Dict, Any, AsyncGenerator
import uuid
import logging
import json
from db.chats import ChatDataRepository
from .llm import LLMService
from .vector import VectorService
from ..models import (
    Chat, Message, ChatResponse, MessageResponse, 
    ChatWithMessages, QueryRequest
)

logger = logging.getLogger(__name__)

class ChatService:
    """Service layer for chat-related operations"""
    
    def __init__(self, llm_service: LLMService, vector_service: VectorService):
        self.chat_repo = ChatDataRepository()
        self.llm_service = llm_service
        self.vector_service = vector_service
    
    async def create_chat(self, user_id: str) -> ChatResponse:
        """Create a new chat for a user"""
        try:
            chat = await self.chat_repo.create_chat(user_id)
            return ChatResponse(
                id=chat.id,
                user_id=chat.user_id,
                created_at=chat.created_at
            )
        except Exception as e:
            logger.error(f"Error creating chat for user {user_id}: {e}")
            raise Exception("Failed to create chat")
    
    async def get_user_chats(self, user_id: str) -> List[ChatResponse]:
        """Get all chats for a user"""
        try:
            chats = await self.chat_repo.get_user_chats(user_id)
            return [
                ChatResponse(
                    id=chat.id,
                    user_id=chat.user_id,
                    created_at=chat.created_at,
                    name=chat.name
                )
                for chat in chats
            ]
        except Exception as e:
            logger.error(f"Error getting chats for user {user_id}: {e}")
            raise Exception("Failed to retrieve chats")
    
    async def get_chat_messages(self, chat_id: uuid.UUID) -> List[MessageResponse]:
        """Get all messages for a chat"""
        try:
            messages = await self.chat_repo.get_chat_messages(chat_id)
            return [
                MessageResponse(
                    id=msg.id,
                    chat_id=msg.chat_id,
                    role=msg.role,
                    content=msg.content,
                    context=msg.context,
                    created_at=msg.created_at
                )
                for msg in messages
            ]
        except Exception as e:
            logger.error(f"Error getting messages for chat {chat_id}: {e}")
            raise Exception("Failed to retrieve messages")
    
    async def get_chat_with_messages(self, chat_id: uuid.UUID, user_id: Optional[str] = None) -> Optional[ChatWithMessages]:
        """Get a chat with all its messages"""
        try:
            chat = await self.chat_repo.get_chat_with_messages(chat_id, user_id)
            if not chat:
                return None
            
            messages = [
                MessageResponse(
                    id=msg.id,
                    chat_id=msg.chat_id,
                    role=msg.role,
                    content=msg.content,
                    context=msg.context,
                    created_at=msg.created_at
                )
                for msg in chat.messages
            ]
            
            return ChatWithMessages(
                id=chat.id,
                user_id=chat.user_id,
                created_at=chat.created_at,
                messages=messages
            )
        except Exception as e:
            logger.error(f"Error getting chat {chat_id} with messages: {e}")
            raise Exception("Failed to retrieve chat with messages")
    
    async def get_or_create_chat(self, user_id: str, chat_id: Optional[uuid.UUID] = None) -> Chat:
        """Get existing chat or create new one"""
        try:
            if chat_id:
                chat = await self.chat_repo.get_chat(chat_id, user_id)
                if not chat:
                    raise ValueError("Chat not found")
                return chat
            else:
                return await self.chat_repo.create_chat(user_id)
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error getting or creating chat: {e}")
            raise Exception("Failed to get or create chat")
    
    async def create_message(
        self, 
        chat_id: uuid.UUID, 
        role: str, 
        content: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> MessageResponse:
        """Create a new message in a chat"""
        try:
            message = await self.chat_repo.create_message(chat_id, role, content, context)
            return MessageResponse(
                id=message.id,
                chat_id=message.chat_id,
                role=message.role,
                content=message.content,
                context=message.context,
                created_at=message.created_at
            )
        except Exception as e:
            logger.error(f"Error creating message in chat {chat_id}: {e}")
            raise Exception("Failed to create message")
    
    async def process_query_stream(self, request: QueryRequest, chat: Chat, user_custom_prompt: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a query with streaming response"""
        try:
            # 1. Save user message
            await self.create_message(chat.id, "user", request.query)
            
            # 2. Get chat message history (excluding the just-saved user message for now)
            messages = await self.chat_repo.get_chat_messages(chat.id)
            # Convert to format expected by LLM (exclude the current user message)
            chat_history = []
            for msg in messages[:-1]:  # Exclude the last message (current user message)
                chat_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # 3. Get context from vector search
            context = await self.vector_service.search(request.query)
            
            # 4. Stream response
            full_response = ""
            
            # Send start event
            yield {
                "type": "start", 
                "chat_id": str(chat.id)
            }
            
            # Stream LLM response with chat history and custom system prompt
            async for chunk in self.llm_service.stream_with_context(
                request.query, 
                context, 
                chat_history, 
                custom_system_prompt=user_custom_prompt
            ):
                full_response += chunk
                yield {
                    "type": "content", 
                    "content": chunk
                }
            
            # Save assistant message
            await self.create_message(
                chat.id, 
                "assistant", 
                full_response,
                {"retrieved_chunks": context}
            )
            
            # Generate and update chat name if this is a new chat without a name
            if not chat.name:
                try:
                    chat_name = await self.llm_service.generate_chat_name(request.query, full_response)
                    await self.update_chat_name(chat.id, chat_name)
                    chat.name = chat_name  # Update local object
                except Exception as e:
                    logger.error(f"Failed to generate chat name: {e}")
            
            # Send end event
            yield {
                "type": "end", 
                "content": full_response,
                "chat_name": chat.name
            }
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            yield {
                "type": "error", 
                "error": str(e)
            }

    async def update_chat_name(self, chat_id: uuid.UUID, name: str) -> bool:
        """Update chat name"""
        try:
            success = await self.chat_repo.update_chat_name(chat_id, name)
            if success:
                logger.info(f"Updated chat {chat_id} name to: {name}")
            return success
        except Exception as e:
            logger.error(f"Error updating chat name for {chat_id}: {e}")
            raise Exception("Failed to update chat name") 