from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from typing import List, Optional
import uuid
import json
import logging

from .models import (
    QueryRequest, UserCreate, UserResponse, ChatResponse, 
    MessageResponse, ChatWithMessages, TranscriptionResponse,
    CustomPromptRequest, CustomPromptResponse
)
from .services import LLMService, VectorService, ChatService, UserService, TranscriptionService
from .dependencies import auth_middleware

logger = logging.getLogger(__name__)

router = APIRouter()

# Global service instances (will be injected by main.py)
llm_service: Optional[LLMService] = None
vector_service: Optional[VectorService] = None
chat_service: Optional[ChatService] = None
user_service: Optional[UserService] = None
transcription_service: Optional[TranscriptionService] = None

def set_services(llm: LLMService, vector: VectorService):
    """Set the service instances"""
    global llm_service, vector_service, chat_service, user_service, transcription_service
    llm_service = llm
    vector_service = vector
    chat_service = ChatService(llm, vector)
    user_service = UserService()
    transcription_service = TranscriptionService()

@router.post("/users", response_model=UserResponse)
async def create_user(
    user: UserCreate,
    user_id: str = Depends(auth_middleware)
):
    """Create a new user"""
    if not user_service:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    try:
        return await user_service.create_user(user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user")

@router.post("/query")
async def query_endpoint(
    request: QueryRequest,
    user_id: str = Depends(auth_middleware)
):
    """Handle streaming Q&A with vector search context"""
    if not chat_service or not user_service:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    try:
        # 1. Verify user exists - use authenticated user_id
        user_exists = await user_service.user_exists(user_id)
        if not user_exists:
            raise HTTPException(status_code=404, detail="User not found")
        
        # 2. Override request user_id with authenticated user_id for security
        request.user_id = user_id
        
        # 3. Get user to access custom system prompt
        user = await user_service.get_user(user_id)
        user_custom_prompt = user.custom_system_prompt if user else None
        
        # 4. Get or create chat
        chat = await chat_service.get_or_create_chat(user_id, request.chat_id)
        
        # 5. Stream response
        async def generate():
            try:
                async for event in chat_service.process_query_stream(request, chat, user_custom_prompt):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                logger.error(f"Error in streaming: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        
        return EventSourceResponse(generate())
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/chats", response_model=List[ChatResponse])
async def get_user_chats(
    user_id: str = Depends(auth_middleware)
):
    """Get all chats for a user"""
    if not chat_service:
        raise HTTPException(status_code=500, detail="Services not initialized")
    try:
        user_chats = await chat_service.get_user_chats(user_id)
        return user_chats
    except Exception as e:
        logger.error(f"Error getting user chats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chats")

@router.get("/chats/{chat_id}", response_model=ChatWithMessages)
async def get_chat_messages(
    chat_id: uuid.UUID,
    user_id: str = Depends(auth_middleware)
):
    """Get all messages for a chat"""
    if not chat_service:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    try:
        # Verify the chat belongs to the authenticated user
        chat = await chat_service.get_chat_with_messages(chat_id)
        if not chat or str(chat.user_id) != user_id:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        return chat
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat with messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat")

@router.delete("/chats/{chat_id}")
async def delete_chat(
    chat_id: uuid.UUID,
    user_id: str = Depends(auth_middleware)
):
    """Delete a chat and all its messages"""
    if not chat_service:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    try:
        # Verify the chat belongs to the authenticated user
        chat = await chat_service.get_chat_with_messages(chat_id)
        if not chat or str(chat.user_id) != user_id:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        success = await chat_service.chat_repo.delete_chat(chat_id)
        if not success:
            raise HTTPException(status_code=404, detail="Chat not found")
        return {"message": "Chat deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete chat")

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    user_id: str = Depends(auth_middleware)
):
    if not transcription_service:
        raise HTTPException(status_code=500, detail="Transcription service not initialized")
    
    # Validate file type (optional - Whisper supports many formats)
    allowed_types = {
        "audio/wav", "audio/mp3", "audio/mpeg", "audio/mp4", "audio/m4a", 
        "audio/webm", "audio/ogg", "audio/flac", "audio/aac"
    }
    
    if audio_file.content_type and audio_file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported audio format. Supported formats: {', '.join(allowed_types)}"
        )
    
    try:
        # Transcribe the audio
        transcription = await transcription_service.transcribe_audio(
            audio_file.file, 
            audio_file.filename or "audio.wav"
        )
        
        return TranscriptionResponse(transcription=transcription)
        
    except Exception as e:
        logger.error(f"Error in transcription endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@router.post("/users/custom-prompt", response_model=CustomPromptResponse)
async def update_custom_system_prompt(
    request: CustomPromptRequest,
    user_id: str = Depends(auth_middleware)
):
    """Update user's custom system prompt"""
    if not user_service:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    try:
        return await user_service.update_custom_system_prompt(user_id, request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating custom system prompt: {e}")
        raise HTTPException(status_code=500, detail="Failed to update custom system prompt")

@router.get("/users/custom-prompt", response_model=CustomPromptResponse)
async def get_custom_system_prompt(
    user_id: str = Depends(auth_middleware)
):
    """Get user's custom system prompt"""
    if not user_service:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    try:
        return await user_service.get_custom_system_prompt(user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting custom system prompt: {e}")
        raise HTTPException(status_code=500, detail="Failed to get custom system prompt")

@router.post("/users/custom-prompt/reset", response_model=CustomPromptResponse)
async def reset_custom_system_prompt(
    user_id: str = Depends(auth_middleware)
):
    """Reset user's custom system prompt to default"""
    if not user_service:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    try:
        # Reset by passing None as custom prompt
        request = CustomPromptRequest(custom_system_prompt=None)
        return await user_service.update_custom_system_prompt(user_id, request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error resetting custom system prompt: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset custom system prompt")

