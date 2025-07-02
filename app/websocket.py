from fastapi import WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.routing import APIRouter
from typing import Dict, Optional
import json
import logging
from clerk_backend_api.jwks_helpers import verify_token, VerifyTokenOptions
from .dependencies import auth_middleware
from .services import LLMService, VectorService, ChatService, UserService, ElevenLabsService
from .config import settings
import asyncio


logger = logging.getLogger(__name__)

router = APIRouter()

# Global service instances (will be injected by main.py)
llm_service: Optional[LLMService] = None
vector_service: Optional[VectorService] = None
chat_service: Optional[ChatService] = None
user_service: Optional[UserService] = None
elevenlabs_service: Optional[ElevenLabsService] = None

def set_websocket_services(llm: LLMService, vector: VectorService, chat: ChatService, user: UserService):
    """Set the service instances for WebSocket"""
    global llm_service, vector_service, chat_service, user_service, elevenlabs_service
    llm_service = llm
    vector_service = vector
    chat_service = chat
    user_service = user
    elevenlabs_service = ElevenLabsService()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        self.active_connections[user_id] = websocket
        logger.info(f"WebSocket connected for user: {user_id}")

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"WebSocket disconnected for user: {user_id}")

    async def send_message(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to user {user_id}: {e}")
                # Remove the connection if sending fails
                self.disconnect(user_id)

manager = ConnectionManager()

async def authenticate_websocket(token: str) -> str:
    
    try:
        claims = verify_token(
            token,
            VerifyTokenOptions(
                authorized_parties=["http://localhost:3000", "https://russell.hfgok.com"],
                secret_key=settings.CLERK_SECRET_KEY
            )
        )
        return claims.get("sub")
    except Exception as e:
        logger.error(f"WebSocket authentication failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat with streaming LLM responses"""
    user_id = None
    
    try:
        await websocket.accept()
        
        # Expect first message to be authentication
        auth_data = await websocket.receive_text()
        
        auth_message = json.loads(auth_data)
        
        if auth_message.get("type") != "auth":
            await websocket.send_text(json.dumps({
                "type": "error", 
                "error": "First message must be authentication"
            }))
            await websocket.close()
            return
        
        # Authenticate
        token = auth_message.get("token")
        if not token:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "error": "Token required"
            }))
            await websocket.close()
            return
        
        user_id = await authenticate_websocket(token)
        
        await manager.connect(websocket, user_id)
        
        # Verify user exists using user service
        if not await user_service.user_exists(user_id):
            await manager.send_message(user_id, {
                "type": "error",
                "error": "User not found"
            })
            await websocket.close()
            return
        
        # Send authentication success
        await manager.send_message(user_id, {
            "type": "auth_success",
            "user_id": user_id
        })
        
        # Handle chat messages
        while True:
            try:
                data = await websocket.receive_text()
                
                message = json.loads(data)
                
                if message.get("type") == "chat":
                    logger.info(f"Processing chat message for user {user_id}")
                    await handle_chat_message(user_id, message)
                    logger.info(f"Completed chat message processing for user {user_id}")
                else:
                    await manager.send_message(user_id, {
                        "type": "error",
                        "error": "Unknown message type"
                    })
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnect detected for user {user_id}")
                break
            except json.JSONDecodeError:
                logger.error(f"JSON decode error for user {user_id}")
                await manager.send_message(user_id, {
                    "type": "error",
                    "error": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"Error handling WebSocket message for user {user_id}: {e}")
                await manager.send_message(user_id, {
                    "type": "error",
                    "error": "Internal server error"
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        if user_id:
            manager.disconnect(user_id)

@router.websocket("/ws/test")
async def websocket_test_endpoint(websocket: WebSocket):
    """Test WebSocket endpoint with full LLM streaming without authentication"""
    await websocket.accept()
    logger.info("Test WebSocket connection established")
    
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Test WebSocket connected. Send a message to test full LLM streaming."
        }))
        
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                query = message_data.get("message", "Hello from test endpoint!")
                
                logger.info(f"Test endpoint received: {query}")
                
                # Check if services are available
                if not llm_service or not vector_service:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": "Services not initialized"
                    }))
                    continue
                
                # Send start message
                await websocket.send_text(json.dumps({
                    "type": "stream_start",
                    "query": query
                }))
                
                # Get context from vector search (same as real flow)
                
                context = await vector_service.search(query)
                
                
                
                # Stream LLM response (same as real flow)
                
                full_response = ""
                
                try:
                    async for chunk in llm_service.stream_with_context(query, context, []):
                        full_response += chunk
                        await websocket.send_text(json.dumps({
                            "type": "chunk",
                            "content": chunk
                        }))
                        
                        logger.debug(f"Sent chunk: {chunk[:50]}...")
                        
                except Exception as e:
                    logger.error(f"Error during LLM streaming in test: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": f"Streaming error: {str(e)}"
                    }))
                    continue
                
                # Send completion
                await websocket.send_text(json.dumps({
                    "type": "complete",
                    "full_response": full_response,
                    "context_chunks": len(context)
                }))
                
                logger.info("Test LLM streaming completed successfully")
                
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": "Invalid JSON"
                }))
            except WebSocketDisconnect:
                logger.info("Test WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error in test WebSocket: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "error": str(e)
                }))
                
    except Exception as e:
        logger.error(f"Test WebSocket connection error: {e}")
    finally:
        logger.info("Test WebSocket connection closed")

@router.websocket("/ws/chat/voice")
async def websocket_chat_voice_endpoint(websocket: WebSocket):
    """WebSocket endpoint for chat with full LLM response followed by voice streaming"""
    user_id = None
    
    try:
        await websocket.accept()
        
        # Expect first message to be authentication
        auth_data = await websocket.receive_text()
        
        auth_message = json.loads(auth_data)
        
        if auth_message.get("type") != "auth":
            await websocket.send_text(json.dumps({
                "type": "error", 
                "error": "First message must be authentication"
            }))
            await websocket.close()
            return
        
        # Authenticate
        token = auth_message.get("token")
        if not token:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "error": "Token required"
            }))
            await websocket.close()
            return
        
        user_id = await authenticate_websocket(token)
        
        await manager.connect(websocket, user_id)
        
        # Verify user exists using user service
        if not await user_service.user_exists(user_id):
            await manager.send_message(user_id, {
                "type": "error",
                "error": "User not found"
            })
            await websocket.close()
            return
        
        # Send authentication success
        await manager.send_message(user_id, {
            "type": "auth_success",
            "user_id": user_id
        })
        
        # Handle chat messages
        while True:
            try:
                data = await websocket.receive_text()
                
                message = json.loads(data)
                
                if message.get("type") == "chat":
                    logger.info(f"Processing voice chat message for user {user_id}")
                    await handle_voice_chat_message(user_id, message)
                    logger.info(f"Completed voice chat message processing for user {user_id}")
                else:
                    await manager.send_message(user_id, {
                        "type": "error",
                        "error": "Unknown message type"
                    })
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnect detected for user {user_id}")
                break
            except json.JSONDecodeError:
                logger.error(f"JSON decode error for user {user_id}")
                await manager.send_message(user_id, {
                    "type": "error",
                    "error": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"Error handling WebSocket message for user {user_id}: {e}")
                await manager.send_message(user_id, {
                    "type": "error",
                    "error": "Internal server error"
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        if user_id:
            manager.disconnect(user_id)

@router.websocket("/ws/chat/voice/test")
async def websocket_chat_voice_test_endpoint(websocket: WebSocket):
    """WebSocket endpoint for chat with full LLM response followed by voice streaming - NO AUTH"""
    await websocket.accept()
    logger.info("Voice test WebSocket connection established (no auth)")
    
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Voice test WebSocket connected. Send a message to test full LLM response + voice streaming."
        }))
        
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                query = message_data.get("message", "Hello from voice test endpoint!")
                voice_id = message_data.get("voice_id")  # Optional voice ID override
                
                logger.info(f"Voice test endpoint received: {query}")
                
                # Check if services are available
                if not llm_service or not vector_service or not elevenlabs_service:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": "Services not initialized"
                    }))
                    continue
                
                # Check if voice is available
                if not elevenlabs_service.api_key:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": "Voice service not configured"
                    }))
                    continue
                
                # Send processing start message
                await websocket.send_text(json.dumps({
                    "type": "processing_start",
                    "query": query
                }))
                
                # Get context from vector search
                context = await vector_service.search(query)
                
                # Get full LLM response (not streamed)
                logger.info("Getting full LLM response for test endpoint")
                full_response = ""
                
                try:
                    # Collect the full response without sending chunks
                    async for chunk in llm_service.stream_with_context(query, context, []):
                        full_response += chunk
                        
                except Exception as e:
                    logger.error(f"Error getting LLM response in test: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": f"LLM error: {str(e)}"
                    }))
                    continue
                
                # Send the full text response
                await websocket.send_text(json.dumps({
                    "type": "text_complete",
                    "full_response": full_response,
                    "context_chunks": len(context)
                }))
                
                # Now stream the voice
                logger.info("Starting voice streaming for test endpoint")
                await websocket.send_text(json.dumps({
                    "type": "voice_start"
                }))
                
                try:
                    # Stream voice for the full response
                    async for audio_chunk in elevenlabs_service.text_to_speech_stream(full_response, voice_id=voice_id):
                        audio_base64 = elevenlabs_service.encode_audio_base64(audio_chunk)
                        await websocket.send_text(json.dumps({
                            "type": "voice_chunk",
                            "audio": audio_base64,
                            "format": "mp3"
                        }))
                    
                    await websocket.send_text(json.dumps({
                        "type": "voice_complete"
                    }))
                    
                    logger.info("Voice test streaming completed successfully")
                    
                except Exception as e:
                    logger.error(f"Error streaming voice in test: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "voice_error",
                        "error": str(e)
                    }))
                
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": "Invalid JSON"
                }))
            except WebSocketDisconnect:
                logger.info("Voice test WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error in voice test WebSocket: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "error": str(e)
                }))
                
    except Exception as e:
        logger.error(f"Voice test WebSocket connection error: {e}")
    finally:
        logger.info("Voice test WebSocket connection closed")

async def handle_chat_message(user_id: str, message: dict):
    
    if not llm_service or not vector_service or not chat_service:
        await manager.send_message(user_id, {
            "type": "error",
            "error": "Services not initialized"
        })
        return
    
    try:
        query = message.get("message")
        chat_id = message.get("chat_id")
        enable_voice = message.get("enable_voice", False)  # Client can request voice
        
        if not query:
            await manager.send_message(user_id, {
                "type": "error",
                "error": "Message content required"
            })
            return
        
        # Convert chat_id to UUID if provided
        chat_uuid = None
        if chat_id:
            try:
                import uuid
                chat_uuid = uuid.UUID(chat_id)
            except ValueError:
                await manager.send_message(user_id, {
                    "type": "error",
                    "error": "Invalid chat ID format"
                })
                return
        
        # Get or create chat using chat service
        chat = await chat_service.get_or_create_chat(user_id, chat_uuid)
        if not chat:
            await manager.send_message(user_id, {
                "type": "error",
                "error": "Chat not found"
            })
            return
        
        # Get user
        user = await user_service.get_user(user_id)
        
        # Save user message using chat service
        user_message = await chat_service.create_message(chat.id, "user", query)
        
        # Send chat start message with voice status
        voice_enabled = enable_voice and elevenlabs_service and elevenlabs_service.api_key is not None
        await manager.send_message(user_id, {
            "type": "chat_start",
            "chat_id": str(chat.id),
            "message_id": str(user_message.id),
            "voice_enabled": voice_enabled
        })
        
        # Get chat message history (excluding the just-saved user message)
        messages = await chat_service.get_chat_messages(chat.id)
        chat_history = []
        for msg in messages[:-1]:  # Exclude the last message (current user message)
            chat_history.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Get context from vector search
        context = await vector_service.search(query)
        
        # Stream LLM response
        full_response = ""
        sentence_buffer = ""
        
        try:
            async for chunk in llm_service.stream_with_context(query, context, chat_history):
                full_response += chunk
                sentence_buffer += chunk
                
                await manager.send_message(user_id, {
                    "type": "chat_chunk",
                    "content": chunk,
                    "chat_id": str(chat.id)
                })
                
                # Check if connection is still active after each chunk
                if user_id not in manager.active_connections:
                    logger.warning(f"Connection lost for user {user_id} during streaming")
                    return
                
                # Process voice synthesis for complete sentences
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
                            # Send voice start
                            await manager.send_message(user_id, {
                                "type": "voice_start",
                                "text": complete_sentence,
                                "chat_id": str(chat.id)
                            })
                            
                            try:
                                # Stream voice chunks
                                async for audio_chunk in elevenlabs_service.text_to_speech_stream(complete_sentence):
                                    audio_base64 = elevenlabs_service.encode_audio_base64(audio_chunk)
                                    await manager.send_message(user_id, {
                                        "type": "voice_chunk",
                                        "audio": audio_base64,
                                        "format": "mp3",
                                        "chat_id": str(chat.id)
                                    })
                                
                                await manager.send_message(user_id, {
                                    "type": "voice_end",
                                    "text": complete_sentence,
                                    "chat_id": str(chat.id)
                                })
                            except Exception as e:
                                logger.error(f"Error synthesizing voice: {e}")
                                await manager.send_message(user_id, {
                                    "type": "voice_error",
                                    "error": str(e),
                                    "text": complete_sentence,
                                    "chat_id": str(chat.id)
                                })
                                
        except Exception as e:
            logger.error(f"Error during LLM streaming for user {user_id}: {e}")
            await manager.send_message(user_id, {
                "type": "error",
                "error": "Streaming interrupted"
            })
            return
        
        # Process any remaining text in voice buffer
        if voice_enabled and sentence_buffer.strip():
            await manager.send_message(user_id, {
                "type": "voice_start",
                "text": sentence_buffer.strip(),
                "chat_id": str(chat.id)
            })
            
            try:
                async for audio_chunk in elevenlabs_service.text_to_speech_stream(sentence_buffer.strip()):
                    audio_base64 = elevenlabs_service.encode_audio_base64(audio_chunk)
                    await manager.send_message(user_id, {
                        "type": "voice_chunk",
                        "audio": audio_base64,
                        "format": "mp3",
                        "chat_id": str(chat.id)
                    })
                
                await manager.send_message(user_id, {
                    "type": "voice_end",
                    "text": sentence_buffer.strip(),
                    "chat_id": str(chat.id)
                })
            except Exception as e:
                logger.error(f"Error synthesizing remaining voice: {e}")
                await manager.send_message(user_id, {
                    "type": "voice_error",
                    "error": str(e),
                    "text": sentence_buffer.strip(),
                    "chat_id": str(chat.id)
                })
        
        # Save assistant message using chat service
        assistant_message = await chat_service.create_message(
            chat.id, 
            "assistant",
            full_response, 
            {"retrieved_chunks": context}
        )
        
        # Generate and save chat name for new chats
        chat_name = None
        if not chat.name:
            try:
                chat_name = await llm_service.generate_chat_name(query, full_response)
                await chat_service.update_chat_name(chat.id, chat_name)
                logger.info(f"Generated chat name for {chat.id}: {chat_name}")
            except Exception as e:
                logger.error(f"Failed to generate chat name for {chat.id}: {e}")
                chat_name = None
        else:
            chat_name = chat.name
        
        # Send completion message
        await manager.send_message(user_id, {
            "type": "chat_complete",
            "chat_id": str(chat.id),
            "message_id": str(assistant_message.id),
            "full_response": full_response,
            "chat_name": chat_name
        })
        
    except Exception as e:
        logger.error(f"Error handling chat message: {e}")
        await manager.send_message(user_id, {
            "type": "error",
            "error": f"Failed to process message: {str(e)}"
        })

async def handle_voice_chat_message(user_id: str, message: dict):
    """Handle chat message with full LLM response first, then voice streaming"""
    
    if not llm_service or not vector_service or not chat_service or not elevenlabs_service:
        await manager.send_message(user_id, {
            "type": "error",
            "error": "Services not initialized"
        })
        return
    
    try:
        query = message.get("message")
        chat_id = message.get("chat_id")
        voice_id = message.get("voice_id")  # Optional voice ID override
        
        if not query:
            await manager.send_message(user_id, {
                "type": "error",
                "error": "Message content required"
            })
            return
        
        # Check if voice is available
        if not elevenlabs_service.api_key:
            await manager.send_message(user_id, {
                "type": "error",
                "error": "Voice service not configured"
            })
            return
        
        # Convert chat_id to UUID if provided
        chat_uuid = None
        if chat_id:
            try:
                import uuid
                chat_uuid = uuid.UUID(chat_id)
            except ValueError:
                await manager.send_message(user_id, {
                    "type": "error",
                    "error": "Invalid chat ID format"
                })
                return
        
        # Get or create chat using chat service
        chat = await chat_service.get_or_create_chat(user_id, chat_uuid)
        if not chat:
            await manager.send_message(user_id, {
                "type": "error",
                "error": "Chat not found"
            })
            return
        
        # Get user
        user = await user_service.get_user(user_id)
        
        # Save user message using chat service
        user_message = await chat_service.create_message(chat.id, "user", query)
        
        # Send processing start message
        await manager.send_message(user_id, {
            "type": "processing_start",
            "chat_id": str(chat.id),
            "message_id": str(user_message.id)
        })
        
        # Get chat message history (excluding the just-saved user message)
        messages = await chat_service.get_chat_messages(chat.id)
        chat_history = []
        for msg in messages[:-1]:  # Exclude the last message (current user message)
            chat_history.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Get context from vector search
        context = await vector_service.search(query)
        
        # Get full LLM response (not streamed)
        logger.info(f"Getting full LLM response for user {user_id}")
        full_response = ""
        
        try:
            # Collect the full response without sending chunks
            async for chunk in llm_service.stream_with_context(query, context, chat_history):
                full_response += chunk
                
        except Exception as e:
            logger.error(f"Error getting LLM response for user {user_id}: {e}")
            await manager.send_message(user_id, {
                "type": "error",
                "error": "Failed to generate response"
            })
            return
        
        # Save assistant message
        assistant_message = await chat_service.create_message(
            chat.id, 
            "assistant",
            full_response, 
            {"retrieved_chunks": context}
        )
        
        # Generate and save chat name for new chats
        chat_name = None
        if not chat.name:
            try:
                chat_name = await llm_service.generate_chat_name(query, full_response)
                await chat_service.update_chat_name(chat.id, chat_name)
                logger.info(f"Generated chat name for {chat.id}: {chat_name}")
            except Exception as e:
                logger.error(f"Failed to generate chat name for {chat.id}: {e}")
                chat_name = None
        else:
            chat_name = chat.name
        
        # Send the full text response
        await manager.send_message(user_id, {
            "type": "text_complete",
            "chat_id": str(chat.id),
            "message_id": str(assistant_message.id),
            "full_response": full_response,
            "chat_name": chat_name
        })
        
        # Now stream the voice
        logger.info(f"Starting voice streaming for user {user_id}")
        await manager.send_message(user_id, {
            "type": "voice_start",
            "chat_id": str(chat.id)
        })
        
        try:
            # Stream voice for the full response
            async for audio_chunk in elevenlabs_service.text_to_speech_stream(full_response, voice_id=voice_id):
                audio_base64 = elevenlabs_service.encode_audio_base64(audio_chunk)
                await manager.send_message(user_id, {
                    "type": "voice_chunk",
                    "audio": audio_base64,
                    "format": "mp3",
                    "chat_id": str(chat.id)
                })
            
            await manager.send_message(user_id, {
                "type": "voice_complete",
                "chat_id": str(chat.id)
            })
            
        except Exception as e:
            logger.error(f"Error streaming voice: {e}")
            await manager.send_message(user_id, {
                "type": "voice_error",
                "error": str(e),
                "chat_id": str(chat.id)
            })
        
    except Exception as e:
        logger.error(f"Error handling voice chat message: {e}")
        await manager.send_message(user_id, {
            "type": "error",
            "error": f"Failed to process message: {str(e)}"
        }) 