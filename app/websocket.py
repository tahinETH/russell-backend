from fastapi import WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.routing import APIRouter
from typing import Dict, Optional
import json
import logging
from clerk_backend_api.jwks_helpers import verify_token, VerifyTokenOptions
from .dependencies import auth_middleware
from .services import LLMService, VectorService, ChatService, UserService
from .config import settings
import asyncio


logger = logging.getLogger(__name__)

router = APIRouter()

# Global service instances (will be injected by main.py)
llm_service: Optional[LLMService] = None
vector_service: Optional[VectorService] = None
chat_service: Optional[ChatService] = None
user_service: Optional[UserService] = None

def set_websocket_services(llm: LLMService, vector: VectorService, chat: ChatService, user: UserService):
    """Set the service instances for WebSocket"""
    global llm_service, vector_service, chat_service, user_service
    llm_service = llm
    vector_service = vector
    chat_service = chat
    user_service = user

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
    """Authenticate WebSocket connection using JWT token"""
    try:
        claims = verify_token(
            token,
            VerifyTokenOptions(
                authorized_parties=["http://localhost:3000"],
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
                logger.info("Getting vector context...")
                context = await vector_service.search(query)
                logger.info(f"Got {len(context)} context chunks")
                
                # Stream LLM response (same as real flow)
                logger.info("Starting LLM streaming...")
                full_response = ""
                
                try:
                    async for chunk in llm_service.stream_with_context(query, context):
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
        
        # Save user message using chat service
        user_message = await chat_service.create_message(chat.id, "user", query)
        
        # Send chat start message
        await manager.send_message(user_id, {
            "type": "chat_start",
            "chat_id": str(chat.id),
            "message_id": str(user_message.id)
        })
        
        # Get context from vector search
        context = await vector_service.search(query)
        
        # Stream LLM response
        full_response = ""
        
        try:
            async for chunk in llm_service.stream_with_context(query, context):
                full_response += chunk
                await manager.send_message(user_id, {
                    "type": "chat_chunk",
                    "content": chunk,
                    "chat_id": str(chat.id)
                })
                
                # Check if connection is still active after each chunk
                if user_id not in manager.active_connections:
                    logger.warning(f"Connection lost for user {user_id} during streaming")
                    return
                    
        except Exception as e:
            logger.error(f"Error during LLM streaming for user {user_id}: {e}")
            await manager.send_message(user_id, {
                "type": "error",
                "error": "Streaming interrupted"
            })
            return
        
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