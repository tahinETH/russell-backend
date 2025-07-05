from typing import Optional, List, Dict, Any, AsyncGenerator
import uuid
import logging
import json
import asyncio
from db.chats import ChatDataRepository
from .llm import LLMService
from .vector import VectorService
from .elevenlabs_service import ElevenLabsService
from ..models import (
    Chat, Message, MessageImage, ChatResponse, MessageResponse, MessageImageResponse,
    ChatWithMessages, QueryRequest
)

logger = logging.getLogger(__name__)

class ChatService:
    """Service layer for chat-related operations"""
    
    def __init__(self, llm_service: LLMService, vector_service: VectorService):
        self.chat_repo = ChatDataRepository()
        self.llm_service = llm_service
        self.vector_service = vector_service
        self.elevenlabs_service = ElevenLabsService()
    
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
    
    def _build_message_response(self, message: Message) -> MessageResponse:
        """Helper method to build MessageResponse with images"""
        images = []
        try:
            # Only try to access images if the relationship is loaded
            if hasattr(message, 'images') and message.images is not None:
                images = [
                    MessageImageResponse(
                        id=img.id,
                        message_id=img.message_id,
                        prompt=img.prompt,
                        image_url=img.image_url,
                        created_at=img.created_at
                    )
                    for img in message.images
                ]
        except Exception as e:
            # If there's an issue accessing images (e.g., lazy loading), just log and continue
            logger.warning(f"Could not load images for message {message.id}: {e}")
            images = []
        
        return MessageResponse(
            id=message.id,
            chat_id=message.chat_id,
            role=message.role,
            content=message.content,
            context=message.context,
            created_at=message.created_at,
            images=images
        )
    
    async def get_chat_messages(self, chat_id: uuid.UUID) -> List[MessageResponse]:
        """Get all messages for a chat with their images"""
        try:
            messages = await self.chat_repo.get_chat_messages(chat_id)
            return [self._build_message_response(msg) for msg in messages]
        except Exception as e:
            logger.error(f"Error getting messages for chat {chat_id}: {e}")
            raise Exception("Failed to retrieve messages")
    
    async def get_chat_with_messages(self, chat_id: uuid.UUID, user_id: Optional[str] = None) -> Optional[ChatWithMessages]:
        """Get a chat with all its messages and images"""
        try:
            chat = await self.chat_repo.get_chat_with_messages(chat_id, user_id)
            if not chat:
                return None
            
            messages = [self._build_message_response(msg) for msg in chat.messages]
            
            return ChatWithMessages(
                id=chat.id,
                user_id=chat.user_id,
                created_at=chat.created_at,
                name=chat.name,
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
            # For newly created messages, we know there are no images yet
            return MessageResponse(
                id=message.id,
                chat_id=message.chat_id,
                role=message.role,
                content=message.content,
                context=message.context,
                created_at=message.created_at,
                images=[]  # New messages have no images
            )
        except Exception as e:
            logger.error(f"Error creating message in chat {chat_id}: {e}")
            raise Exception("Failed to create message")
    
    async def create_message_image(
        self,
        message_id: uuid.UUID,
        prompt: str,
        image_url: str
    ) -> MessageImageResponse:
        """Create a new image associated with a message"""
        try:
            image = await self.chat_repo.create_message_image(message_id, prompt, image_url)
            return MessageImageResponse(
                id=image.id,
                message_id=image.message_id,
                prompt=image.prompt,
                image_url=image.image_url,
                created_at=image.created_at
            )
        except Exception as e:
            logger.error(f"Error creating image for message {message_id}: {e}")
            raise Exception("Failed to create message image")
    
    async def process_query_stream(self, request: QueryRequest, chat: Chat) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a query with response and voice synthesis"""
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
            
            # 3. Get context from vector search (skip if this is a lesson mode)
            context = []
            if not request.lesson:  # Only do vector search if not in lesson mode
                context = await self.vector_service.search(request.query)
            
            # 4. Stream response
            full_response = ""
            sentence_buffer = ""
            
            # Send start event with lesson info
            yield {
                "type": "start", 
                "chat_id": str(chat.id),
                "lesson": request.lesson
            }
            
            # Check if voice synthesis is enabled
            voice_enabled = self.elevenlabs_service.api_key is not None
            if voice_enabled:
                yield {
                    "type": "voice_enabled",
                    "enabled": True
                }
            
            # Stream LLM response with chat history and lesson parameter
            async for chunk in self.llm_service.stream_with_context(
                request.query, 
                context, 
                chat_history,
                lesson=request.lesson
            ):
                full_response += chunk
                sentence_buffer += chunk
                
                # Send text chunk
                yield {
                    "type": "content", 
                    "content": chunk
                }
                
                # Check if we have a complete sentence for voice synthesis
                if voice_enabled and any(end in sentence_buffer for end in ['.', '!', '?', '\n']):
                    # Find the last sentence ending
                    last_end = max(
                        sentence_buffer.rfind('.'),
                        sentence_buffer.rfind('!'),
                        sentence_buffer.rfind('?'),
                        sentence_buffer.rfind('\n')
                    )
                    
                    if last_end > -1:
                        # Extract complete sentence(s)
                        complete_sentence = sentence_buffer[:last_end + 1].strip()
                        sentence_buffer = sentence_buffer[last_end + 1:]
                        
                        if complete_sentence:
                            # Stream voice synthesis directly
                            yield {
                                "type": "voice_start",
                                "text": complete_sentence
                            }
                            
                            try:
                                # Stream audio chunks as they come
                                async for audio_chunk in self.elevenlabs_service.text_to_speech_stream(complete_sentence):
                                    # Encode audio chunk to base64 and send
                                    audio_base64 = self.elevenlabs_service.encode_audio_base64(audio_chunk)
                                    yield {
                                        "type": "voice_chunk",
                                        "audio": audio_base64,
                                        "format": "mp3"
                                    }
                                
                                yield {
                                    "type": "voice_end",
                                    "text": complete_sentence
                                }
                            except Exception as e:
                                logger.error(f"Error synthesizing voice: {e}")
                                yield {
                                    "type": "voice_error",
                                    "error": str(e),
                                    "text": complete_sentence
                                }
            
            # Process any remaining text in sentence buffer
            if voice_enabled and sentence_buffer.strip():
                yield {
                    "type": "voice_start",
                    "text": sentence_buffer.strip()
                }
                
                try:
                    async for audio_chunk in self.elevenlabs_service.text_to_speech_stream(sentence_buffer.strip()):
                        audio_base64 = self.elevenlabs_service.encode_audio_base64(audio_chunk)
                        yield {
                            "type": "voice_chunk",
                            "audio": audio_base64,
                            "format": "mp3"
                        }
                    
                    yield {
                        "type": "voice_end",
                        "text": sentence_buffer.strip()
                    }
                except Exception as e:
                    logger.error(f"Error synthesizing remaining voice: {e}")
                    yield {
                        "type": "voice_error",
                        "error": str(e),
                        "text": sentence_buffer.strip()
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
                "chat_name": chat.name,
                "lesson": request.lesson
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
            return await self.chat_repo.update_chat_name(chat_id, name)
        except Exception as e:
            logger.error(f"Error updating chat name: {e}")
            raise Exception("Failed to update chat name") 