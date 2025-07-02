from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from typing import List, Optional
import uuid
import json
import logging
import time

from .models import (
    QueryRequest, QueryResponse, UserCreate, UserResponse, ChatResponse, ChatWithMessages, TranscriptionResponse
)
from .services import LLMService, VectorService, ChatService, UserService, TranscriptionService, ElevenLabsService
from .dependencies import auth_middleware

logger = logging.getLogger(__name__)

router = APIRouter()

# Global service instances (will be injected by main.py)
llm_service: Optional[LLMService] = None
vector_service: Optional[VectorService] = None
chat_service: Optional[ChatService] = None
user_service: Optional[UserService] = None
transcription_service: Optional[TranscriptionService] = None
elevenlabs_service: Optional[ElevenLabsService] = None

def set_services(llm: LLMService, vector: VectorService):
    """Set the service instances"""
    global llm_service, vector_service, chat_service, user_service, transcription_service, elevenlabs_service
    llm_service = llm
    vector_service = vector
    chat_service = ChatService(llm, vector)
    user_service = UserService()
    transcription_service = TranscriptionService()
    elevenlabs_service = ElevenLabsService()

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
    print(request.query,"\n\n",request.lesson)
    
    
    if not chat_service or not user_service:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    # If voice is enabled, check if voice service is available
    if request.enable_voice and (not elevenlabs_service or not elevenlabs_service.api_key):
        raise HTTPException(status_code=500, detail="Voice service not configured")
    
    try:
        # 1. Verify user exists - use authenticated user_id
        user_exists = await user_service.user_exists(user_id)
        if not user_exists:
            raise HTTPException(status_code=404, detail="User not found")
        
        # 2. Get or create chat
        chat = await chat_service.get_or_create_chat(user_id, request.chat_id)
        
        # 3. Handle voice vs streaming response
        if request.enable_voice:
            # Voice response - return complete response with audio
            return await _handle_voice_query(request, chat)
        else:
            # Streaming response - return SSE stream
            async def generate():
                try:
                    async for event in chat_service.process_query_stream(request, chat):
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

async def _handle_voice_query(request: QueryRequest, chat) -> QueryResponse:
    """Handle voice query and return complete response with audio"""
    start_time = time.time()
    
    try:
        # 1. Save user message to database
        user_message = await chat_service.create_message(chat.id, "user", request.query)
        logger.info(f"Saved user message {user_message.id} to chat {chat.id}")
        
        # 2. Get chat history for context (excluding the just-saved user message)
        messages = await chat_service.get_chat_messages(chat.id)
        chat_history = []
        for msg in messages[:-1]:  # Exclude the last message (current user message)
            chat_history.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # 3. Get context from vector search (skip if this is a lesson mode)
        context = []
        if not request.lesson:  # Only do vector search if not in lesson mode
            context = await vector_service.search(request.query)
        
        # 4. Get full LLM response with chat history and lesson parameter
        full_response = ""
        async for chunk in llm_service.stream_with_context(
            request.query, 
            context, 
            chat_history,
            lesson=request.lesson
        ):
            full_response += chunk
        
        # 5. Generate voice audio
        audio_data = await elevenlabs_service.text_to_speech(
            full_response, 
            voice_id=request.voice_id,
            model_id=request.model_id
        )
        
        # 6. Encode audio to base64
        audio_base64 = elevenlabs_service.encode_audio_base64(audio_data)
        
        # 7. Save assistant message to database
        assistant_message = await chat_service.create_message(
            chat.id, 
            "assistant", 
            full_response,
            {"retrieved_chunks": context, "lesson": request.lesson} if request.lesson else {"retrieved_chunks": context}
        )
        logger.info(f"Saved assistant message {assistant_message.id} to chat {chat.id}")
        
        # 8. Generate and save chat name for new chats
        if not chat.name:
            try:
                chat_name = await llm_service.generate_chat_name(request.query, full_response)
                await chat_service.update_chat_name(chat.id, chat_name)
                chat.name = chat_name  # Update local object
                logger.info(f"Generated and saved chat name for {chat.id}: {chat_name}")
            except Exception as e:
                logger.error(f"Failed to generate chat name for {chat.id}: {e}")
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            text_response=full_response,
            audio_base64=audio_base64,
            audio_format="mp3",
            context_chunks=len(context),
            processing_time=processing_time,
            chat_id=chat.id,
            lesson=request.lesson
        )
        
    except Exception as e:
        logger.error(f"Error in voice query processing: {e}")
        raise HTTPException(status_code=500, detail=f"Voice query failed: {str(e)}")

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

