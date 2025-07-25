from fastapi import WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.routing import APIRouter
from typing import Dict, Optional
import json
import logging
from clerk_backend_api.jwks_helpers import verify_token, VerifyTokenOptions
from .dependencies import auth_middleware
from .services import LLMService, VectorService, ChatService, UserService, ElevenLabsService, FalService, ContextService
from .config import settings
import asyncio


logger = logging.getLogger(__name__)

router = APIRouter()

# Global service instances (will be injected by main.py)
llm_service: Optional[LLMService] = None
customer_support_llm_service: Optional[LLMService] = None
vector_service: Optional[VectorService] = None
chat_service: Optional[ChatService] = None
user_service: Optional[UserService] = None
elevenlabs_service: Optional[ElevenLabsService] = None
fal_service: Optional[FalService] = None
context_service: Optional[ContextService] = None

def set_websocket_services(llm: LLMService, vector: VectorService, chat: ChatService, user: UserService, customer_support_llm: LLMService):
    """Set the service instances for WebSocket"""
    global llm_service, customer_support_llm_service, vector_service, chat_service, user_service, elevenlabs_service, fal_service, context_service
    llm_service = llm
    customer_support_llm_service = customer_support_llm
    vector_service = vector
    chat_service = chat
    user_service = user
    elevenlabs_service = ElevenLabsService()
    fal_service = FalService()
    context_service = ContextService()

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
            websocket = self.active_connections[user_id]
            try:
                # Check if the WebSocket is still connected
                if websocket.client_state.name != "CONNECTED":
                    logger.warning(f"WebSocket for user {user_id} is not connected (state: {websocket.client_state.name})")
                    self.disconnect(user_id)
                    return
                
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to user {user_id}: {e}")
                # Remove the connection if sending fails
                self.disconnect(user_id)

    def is_connected(self, user_id: str) -> bool:
        """Check if a user's WebSocket connection is still active"""
        if user_id not in self.active_connections:
            return False
        
        websocket = self.active_connections[user_id]
        try:
            return websocket.client_state.name == "CONNECTED"
        except:
            # If we can't check the state, assume it's disconnected
            self.disconnect(user_id)
            return False

manager = ConnectionManager()

async def authenticate_websocket(token: str) -> str:
    
    try:
        claims = verify_token(
            token,
            VerifyTokenOptions(
                authorized_parties=["http://localhost:3000", "https://russell.hfgok.com", "https://karseltex-int.com"],
                secret_key=settings.CLERK_SECRET_KEY
            )
        )
        return claims.get("sub")
    except Exception as e:
        logger.error(f"WebSocket authentication failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat with voice-first streaming"""
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
                print("data", data)
                
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
                # Prevent infinite loop on connection error
                if "not connected" in str(e):
                    logger.warning(f"WebSocket for user {user_id} is not connected. Closing loop.")
                    break
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

@router.websocket("/ws/karseltex")
async def websocket_karseltex_endpoint(websocket: WebSocket):
    """Stateless WebSocket endpoint for karseltex-int.com with text-only responses using markdown context"""
    
    try:
        await websocket.accept()
        logger.info("Karseltex WebSocket connection accepted")
        
        # Handle chat messages directly without authentication or user tracking
        while True:
            try:
                data = await websocket.receive_text()
                logger.info("Received karseltex message")
                
                message = json.loads(data)
                
                if message.get("type") == "chat":
                    logger.info("Processing karseltex chat message")
                    await handle_karseltex_stateless_message(websocket, message)
                    logger.info("Completed karseltex chat message processing")
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": "Unknown message type"
                    }))
                    
            except WebSocketDisconnect:
                logger.info("Karseltex WebSocket disconnect detected")
                break
            except json.JSONDecodeError:
                logger.error("JSON decode error for karseltex")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error handling karseltex WebSocket message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": "Internal server error"
                }))
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Karseltex WebSocket connection error: {e}")
    finally:
        logger.info("Karseltex WebSocket connection closed")

async def stream_voice_response(user_id: str, full_text: str, chat_id: str):
    """Generate voice for the complete text response using HTTP request"""
    try:
        logger.info(f"Starting voice generation for user {user_id}")
        
        await manager.send_message(user_id, {
            "type": "voice_start",
            "chat_id": chat_id
        })
        
        # Generate complete audio using HTTP request
        audio_data = await elevenlabs_service.text_to_speech(full_text)
        
        # Check if connection is still active
        if not manager.is_connected(user_id):
            logger.warning(f"Connection lost for user {user_id} during voice generation")
            return
        
        # Send the complete audio as base64
        audio_base64 = elevenlabs_service.encode_audio_base64(audio_data)
        await manager.send_message(user_id, {
            "type": "voice_complete",
            "audio": audio_base64,
            "format": "mp3",
            "chat_id": chat_id
        })
        
        logger.info(f"Voice generation completed for user {user_id}")
        
        # STREAMING VERSION - KEPT FOR FUTURE USE
        # Uncomment the code below and comment out the HTTP version above 
        # if you want to switch back to streaming:
        
        # async for audio_chunk in elevenlabs_service.text_to_speech_stream(full_text):
        #     # Check if connection is still active
        #     if not manager.is_connected(user_id):
        #         logger.warning(f"Connection lost for user {user_id} during voice streaming")
        #         return
        #     
        #     audio_base64 = elevenlabs_service.encode_audio_base64(audio_chunk)
        #     await manager.send_message(user_id, {
        #         "type": "voice_chunk",
        #         "audio": audio_base64,
        #         "format": "mp3",
        #         "chat_id": chat_id
        #     })
        # 
        # await manager.send_message(user_id, {
        #     "type": "voice_complete",
        #     "chat_id": chat_id
        # })
        
    except Exception as e:
        logger.error(f"Error generating voice for user {user_id}: {e}")
        await manager.send_message(user_id, {
            "type": "voice_error",
            "error": str(e),
            "chat_id": chat_id
        })

async def stream_image_response(user_id: str, user_query: str, full_text: str, chat_id: str, assistant_message_id: str, lesson: Optional[str] = None):
    """Generate and stream image based on the AI response"""
    try:
        logger.info(f"Starting image generation for user {user_id}")
        
        # Check if services are available
        if not llm_service or not fal_service or not chat_service:
            logger.warning("Image generation skipped: services not available")
            return
        
        # Check if Fal service is configured
        if not fal_service.api_key:
            logger.info("Image generation skipped: Fal service not configured")
            return
        
        # 1. Generate image prompt using GPT-4o
        image_prompt = await llm_service.generate_image_prompt(user_query, full_text, lesson)

        if not image_prompt:
            logger.warning("Failed to generate image prompt")
            return
        
        # Send image generation start message
        await manager.send_message(user_id, {
            "type": "image_start",
            "chat_id": chat_id,
            "prompt": image_prompt,
            "lesson": lesson
        })
        
        # 2. Stream image generation with Fal
        async for event in fal_service.generate_image_stream(image_prompt):
            # Check if connection is still active
            if not manager.is_connected(user_id):
                logger.warning(f"Connection lost for user {user_id} during image generation")
                return
            
            if event["type"] == "progress":
                await manager.send_message(user_id, {
                    "type": "image_progress",
                    "chat_id": chat_id,
                    "message": event.get("message", "Generating image...")
                })
            elif event["type"] == "complete":
                images = event.get("images", [])
                if images:
                    # Get the first image URL
                    image_url = images[0].get("url", "")
                    
                    # 3. Save image to database
                    try:
                        import uuid
                        message_uuid = uuid.UUID(assistant_message_id)
                        saved_image = await chat_service.create_message_image(
                            message_uuid,
                            image_prompt,
                            image_url
                        )
                        logger.info(f"Saved image {saved_image.id} for message {assistant_message_id}")
                    except Exception as save_error:
                        logger.error(f"Failed to save image to database: {save_error}")
                        # Continue anyway, don't fail the whole process
                    
                    # Send the image completion message
                    await manager.send_message(user_id, {
                        "type": "image_complete",
                        "chat_id": chat_id,
                        "image_url": image_url,
                        "prompt": image_prompt,
                        "lesson": lesson
                    })
                    logger.info(f"Image generation completed for user {user_id}")
                else:
                    await manager.send_message(user_id, {
                        "type": "image_error",
                        "chat_id": chat_id,
                        "error": "No image generated"
                    })
            elif event["type"] == "error":
                await manager.send_message(user_id, {
                    "type": "image_error",
                    "chat_id": chat_id,
                    "error": event.get("error", "Image generation failed")
                })
                
    except Exception as e:
        logger.error(f"Error generating image for user {user_id}: {e}")
        await manager.send_message(user_id, {
            "type": "image_error",
            "chat_id": chat_id,
            "error": str(e)
        })

async def handle_chat_message(user_id: str, message: dict):
    """Handle chat message with voice-first approach: full text first, then concurrent voice and image streaming"""
    
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
        enable_image = message.get("enable_image", False)  # Client can request image generation
        lesson = message.get("lesson")  # Extract lesson parameter
        expertise = message.get("expertise", 3)  # Extract expertise level (1-5, default 3)
        
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
        
        # Check if voice is available and requested
        voice_enabled = enable_voice and elevenlabs_service and elevenlabs_service.api_key is not None
        # Check if image generation is available and requested
        image_enabled = enable_image and fal_service and fal_service.api_key is not None
        
        # Send chat start message with voice and image status
        await manager.send_message(user_id, {
            "type": "chat_start",
            "chat_id": str(chat.id),
            "message_id": str(user_message.id),
            "voice_enabled": voice_enabled,
            "image_enabled": image_enabled,
            "lesson": lesson
        })
        
        # Get chat message history (excluding the just-saved user message)
        messages = await chat_service.get_chat_messages(chat.id)
        chat_history = []
        for msg in messages[:-1]:  # Exclude the last message (current user message)
            chat_history.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Get context from vector search (skip if this is a lesson mode)
        context = []
        if not lesson:  # Only do vector search if not in lesson mode
            context = await vector_service.search(query)
        
        # 1. Get full LLM response (collect all chunks without streaming)
        logger.info(f"Generating full LLM response for user {user_id}")
        full_response = ""
        
        try:
            async for chunk in llm_service.stream_with_context(query, context, chat_history, lesson=lesson, expertise=expertise):
                full_response += chunk
                
                # Check if connection is still active
                if not manager.is_connected(user_id):
                    logger.warning(f"Connection lost for user {user_id} during LLM generation")
                    return
                
        except Exception as e:
            logger.error(f"Error during LLM generation for user {user_id}: {e}")
            await manager.send_message(user_id, {
                "type": "error",
                "error": "Failed to generate response"
            })
            return
        
        # Save assistant message using chat service
        assistant_message = await chat_service.create_message(
            chat.id, 
            "assistant",
            full_response, 
            {"retrieved_chunks": context, "lesson": lesson, "expertise": expertise} if lesson else {"retrieved_chunks": context, "expertise": expertise}
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
        
        # 2. Send complete text immediately
        await manager.send_message(user_id, {
            "type": "text_complete",
            "chat_id": str(chat.id),
            "message_id": str(assistant_message.id),
            "full_response": full_response,
            "chat_name": chat_name,
            "lesson": lesson
        })
        
        # 3. Start concurrent tasks for voice and image generation
        tasks = []
        
        # Start voice streaming (if enabled)
        if voice_enabled:
            voice_task = asyncio.create_task(
                stream_voice_response(user_id, full_response, str(chat.id))
            )
            tasks.append(voice_task)
        
        # Start image generation (if enabled)
        if image_enabled:
            image_task = asyncio.create_task(
                stream_image_response(user_id, query, full_response, str(chat.id), str(assistant_message.id), lesson)
            )
            tasks.append(image_task)
        
        # 4. Wait for all tasks to complete
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error in concurrent tasks for user {user_id}: {e}")
        
        # 5. Send final completion message
        await manager.send_message(user_id, {
            "type": "chat_complete",
            "chat_id": str(chat.id),
            "message_id": str(assistant_message.id),
            "voice_enabled": voice_enabled,
            "image_enabled": image_enabled,
            "lesson": lesson
        })
        
    except Exception as e:
        logger.error(f"Error handling chat message: {e}")
        await manager.send_message(user_id, {
            "type": "error",
            "error": f"Failed to process message: {str(e)}"
        })

async def handle_karseltex_stateless_message(websocket: WebSocket, message: dict):
    """Handle karseltex messages in a stateless manner - no user tracking, no database storage"""
    
    if not customer_support_llm_service or not context_service:
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": "Services not initialized"
        }))
        return
    
    try:
        query = message.get("message")
        chat_history = message.get("history", [])  # Accept chat history from frontend
        
        if not query:
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": "Message content required"
            }))
            return
        
        # Send chat start message
        await websocket.send_text(json.dumps({
            "type": "chat_start",
            "mode": "karseltex"
        }))
        
        # Load markdown context
        logger.info("Loading markdown context for karseltex")
        markdown_context = await context_service.get_context()
        
        # Get context info for logging
        context_info = context_service.get_loaded_files_info()
        logger.info(f"Loaded {context_info['file_count']} markdown files, context size: {context_info['context_size']} chars")
        
        # Generate LLM response with markdown context
        logger.info("Generating LLM response for karseltex")
        full_response = ""
        
        try:
            # Create a system message with the markdown context
            system_context = f"""You are a helpful assistant answering questions about Karseltex company. 
Use the following company information to answer user questions accurately and helpfully.

Company Information:
{markdown_context}

Please answer questions based on this information. If you don't have specific information about something, let the user know politely."""
            
            # Use the customer support LLM service with custom context
            async for chunk in customer_support_llm_service.stream_with_custom_context(query, system_context, chat_history):
                full_response += chunk
                
                # Stream each chunk to the client
                await websocket.send_text(json.dumps({
                    "type": "text_chunk",
                    "chunk": chunk
                }))
                
        except Exception as e:
            logger.error(f"Error during LLM generation for karseltex: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": "Failed to generate response"
            }))
            return
        
        # Send complete text
        await websocket.send_text(json.dumps({
            "type": "text_complete",
            "full_response": full_response,
            "mode": "karseltex",
            "context_info": context_info
        }))
        
        # Send final completion message
        await websocket.send_text(json.dumps({
            "type": "chat_complete",
            "mode": "karseltex"
        }))
        
    except Exception as e:
        logger.error(f"Error handling karseltex message: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": f"Failed to process message: {str(e)}"
        }))

async def handle_karseltex_chat_message(user_id: str, message: dict):
    """Handle karseltex chat message with text-only responses using markdown context"""
    
    if not llm_service or not chat_service or not context_service:
        await manager.send_message(user_id, {
            "type": "error",
            "error": "Services not initialized"
        })
        return
    
    try:
        query = message.get("message")
        chat_id = message.get("chat_id")
        
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
        
        # For public karseltex users, create a temporary chat without database persistence
        # This allows the chat to work without authentication
        if user_id.startswith("karseltex_public_"):
            # Create a temporary chat object
            import uuid
            from datetime import datetime
            
            class TempChat:
                def __init__(self):
                    self.id = chat_uuid or uuid.uuid4()
                    self.name = None
                    self.created_at = datetime.utcnow()
            
            class TempMessage:
                def __init__(self, msg_id):
                    self.id = msg_id
            
            chat = TempChat()
            user_message = TempMessage(uuid.uuid4())
        else:
            # Get or create chat using chat service for authenticated users
            chat = await chat_service.get_or_create_chat(user_id, chat_uuid)
            if not chat:
                await manager.send_message(user_id, {
                    "type": "error",
                    "error": "Chat not found"
                })
                return
            
            # Save user message using chat service
            user_message = await chat_service.create_message(chat.id, "user", query)
        
        # Send chat start message (text-only, no voice/image)
        await manager.send_message(user_id, {
            "type": "chat_start",
            "chat_id": str(chat.id),
            "message_id": str(user_message.id),
            "voice_enabled": False,
            "image_enabled": False,
            "mode": "karseltex"
        })
        
        # Get chat message history (excluding the just-saved user message)
        messages = await chat_service.get_chat_messages(chat.id)
        chat_history = []
        for msg in messages[:-1]:  # Exclude the last message (current user message)
            chat_history.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Load markdown context instead of vector search
        logger.info(f"Loading markdown context for karseltex user {user_id}")
        markdown_context = await context_service.get_context()
        
        # Get context info for logging
        context_info = context_service.get_loaded_files_info()
        logger.info(f"Loaded {context_info['file_count']} markdown files, context size: {context_info['context_size']} chars")
        
        # Generate LLM response with markdown context
        logger.info(f"Generating LLM response for karseltex user {user_id}")
        full_response = ""
        
        try:
            # Create a system message with the markdown context
            system_context = f"""You are a helpful assistant answering questions about Karseltex company. 
Use the following company information to answer user questions accurately and helpfully.

Company Information:
{markdown_context}

Please answer questions based on this information. If you don't have specific information about something, let the user know politely."""
            
            # Use the LLM service with custom context
            async for chunk in llm_service.stream_with_custom_context(query, system_context, chat_history):
                full_response += chunk
                
                # Check if connection is still active
                if not manager.is_connected(user_id):
                    logger.warning(f"Connection lost for karseltex user {user_id} during LLM generation")
                    return
                
        except Exception as e:
            logger.error(f"Error during LLM generation for karseltex user {user_id}: {e}")
            await manager.send_message(user_id, {
                "type": "error",
                "error": "Failed to generate response"
            })
            return
        
        # Save assistant message using chat service
        assistant_message = await chat_service.create_message(
            chat.id, 
            "assistant",
            full_response, 
            {"context_files": context_info['files'], "mode": "karseltex"}
        )
        
        # Generate and save chat name for new chats
        chat_name = None
        if not chat.name:
            try:
                chat_name = await llm_service.generate_chat_name(query, full_response)
                await chat_service.update_chat_name(chat.id, chat_name)
                logger.info(f"Generated chat name for karseltex {chat.id}: {chat_name}")
            except Exception as e:
                logger.error(f"Failed to generate chat name for karseltex {chat.id}: {e}")
                chat_name = None
        else:
            chat_name = chat.name
        
        # Send complete text immediately (text-only response)
        await manager.send_message(user_id, {
            "type": "text_complete",
            "chat_id": str(chat.id),
            "message_id": str(assistant_message.id),
            "full_response": full_response,
            "chat_name": chat_name,
            "mode": "karseltex",
            "context_info": context_info
        })
        
        # Send final completion message (no voice/image tasks)
        await manager.send_message(user_id, {
            "type": "chat_complete",
            "chat_id": str(chat.id),
            "message_id": str(assistant_message.id),
            "voice_enabled": False,
            "image_enabled": False,
            "mode": "karseltex"
        })
        
    except Exception as e:
        logger.error(f"Error handling karseltex chat message: {e}")
        await manager.send_message(user_id, {
            "type": "error",
            "error": f"Failed to process karseltex message: {str(e)}"
        })


