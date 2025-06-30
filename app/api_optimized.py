from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import time
import logging
import asyncio

from .models import QueryRequest, QueryResponse
from .services import LLMService, VectorService, ElevenLabsService, ChatService, UserService
from .utils.performance_monitor import PerformanceMonitor, StreamingMetrics
from .dependencies import auth_middleware

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2")

# Global service instances (will be injected by main.py)
llm_service: Optional[LLMService] = None
vector_service: Optional[VectorService] = None
chat_service: Optional[ChatService] = None
user_service: Optional[UserService] = None
elevenlabs_service: Optional[ElevenLabsService] = None

def set_optimized_services(llm: LLMService, vector: VectorService, chat: ChatService, user: UserService, elevenlabs: ElevenLabsService):
    """Set the service instances for optimized API"""
    global llm_service, vector_service, chat_service, user_service, elevenlabs_service
    llm_service = llm
    vector_service = vector
    chat_service = chat
    user_service = user
    elevenlabs_service = elevenlabs

@router.post("/query-optimized", response_model=QueryResponse)
async def optimized_query_endpoint(
    request: QueryRequest,
    user_id: str = Depends(auth_middleware)
):
    """Optimized query endpoint with performance monitoring and chat saving"""
    
    # Check if services are initialized
    if not all([llm_service, vector_service, chat_service, user_service, elevenlabs_service]):
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    # If voice is enabled, check if voice service is available
    if request.enable_voice and not elevenlabs_service.api_key:
        raise HTTPException(status_code=500, detail="Voice service not configured")
    
    try:
        # 1. Verify user exists
        user_exists = await user_service.user_exists(user_id)
        if not user_exists:
            raise HTTPException(status_code=404, detail="User not found")
        
        # 2. Get user to access custom system prompt
        user = await user_service.get_user(user_id)
        user_custom_prompt = user.custom_system_prompt if user else None
        
        # 3. Get or create chat
        chat = await chat_service.get_or_create_chat(user_id, request.chat_id)
        
        # 4. Handle optimized voice query
        if request.enable_voice:
            return await handle_voice_query_optimized(
                request=request,
                chat=chat,
                user_custom_prompt=user_custom_prompt,
                llm_service=llm_service,
                vector_service=vector_service,
                elevenlabs_service=elevenlabs_service,
                chat_service=chat_service
            )
        else:
            # For non-voice queries, we can still use the optimized processing
            # but without audio generation
            request.enable_voice = False
            return await handle_voice_query_optimized(
                request=request,
                chat=chat,
                user_custom_prompt=user_custom_prompt,
                llm_service=llm_service,
                vector_service=vector_service,
                elevenlabs_service=elevenlabs_service,
                chat_service=chat_service
            )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in optimized query endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def handle_voice_query_optimized(
    request: QueryRequest,
    chat,
    user_custom_prompt: Optional[str],
    llm_service: LLMService,
    vector_service: VectorService,
    elevenlabs_service: ElevenLabsService,
    chat_service: ChatService
) -> QueryResponse:
    """Optimized voice query handler with performance monitoring and chat saving"""
    
    monitor = PerformanceMonitor()
    monitor.start_timer("total_processing")
    
    try:
        # 1. Save user message to database
        monitor.start_timer("save_user_message")
        user_message = await chat_service.create_message(chat.id, "user", request.query)
        monitor.end_timer("save_user_message")
        logger.info(f"Saved user message {user_message.id} to chat {chat.id}")
        
        # 2. Get chat history for context (excluding the just-saved user message)
        monitor.start_timer("get_chat_history")
        messages = await chat_service.get_chat_messages(chat.id)
        chat_history = []
        for msg in messages[:-1]:  # Exclude the last message (current user message)
            chat_history.append({
                "role": msg.role,
                "content": msg.content
            })
        monitor.end_timer("get_chat_history")
        
        # 3. Start vector search immediately (non-blocking)
        monitor.start_timer("vector_search")
        context_task = asyncio.create_task(vector_service.search(request.query))
        
        # 4. Prepare for LLM streaming
        full_response = ""
        llm_metrics = StreamingMetrics()
        
        # 5. Wait for context (should be quick)
        context = await context_task
        monitor.end_timer("vector_search")
        
        # 6. Stream LLM response with metrics and chat history
        monitor.start_timer("llm_generation")
        async for chunk in llm_service.stream_with_context(
            request.query, 
            context, 
            chat_history,  # Include chat history for context
            user_custom_prompt
        ):
            full_response += chunk
            llm_metrics.record_chunk(len(chunk))
        
        monitor.end_timer("llm_generation")
        llm_stats = llm_metrics.get_stats()
        logger.info(f"LLM Stats: Time to first chunk: {llm_stats['time_to_first_chunk']:.3f}s, "
                   f"Total chunks: {llm_stats['chunk_count']}")
        
        # 7. Generate voice audio with optimized settings (only if voice is enabled)
        audio_base64 = None
        if request.enable_voice:
            monitor.start_timer("voice_synthesis")
            
            # Use streaming if text is long enough
            if len(full_response) > 200:  # Arbitrary threshold
                # For longer texts, use streaming with higher optimization
                audio_chunks = []
                tts_metrics = StreamingMetrics()
                
                async for audio_chunk in elevenlabs_service.text_to_speech_stream(
                    full_response,
                    voice_id=request.voice_id,
                    model_id="eleven_flash_v2_5",
                    optimize_streaming_latency=4  # Maximum optimization
                ):
                    audio_chunks.append(audio_chunk)
                    tts_metrics.record_chunk(len(audio_chunk))
                
                audio_data = b''.join(audio_chunks)
                
                tts_stats = tts_metrics.get_stats()
                logger.info(f"TTS Stats: Time to first chunk: {tts_stats['time_to_first_chunk']:.3f}s, "
                           f"Audio chunks: {tts_stats['chunk_count']}")
            else:
                # For shorter texts, use regular synthesis
                audio_data = await elevenlabs_service.text_to_speech(
                    full_response,
                    voice_id=request.voice_id,
                    model_id=request.model_id
                )
            
            monitor.end_timer("voice_synthesis")
            
            # 8. Encode audio
            monitor.start_timer("audio_encoding")
            audio_base64 = elevenlabs_service.encode_audio_base64(audio_data)
            monitor.end_timer("audio_encoding")
        
        # 9. Save assistant message to database
        monitor.start_timer("save_assistant_message")
        assistant_message = await chat_service.create_message(
            chat.id, 
            "assistant", 
            full_response,
            {"retrieved_chunks": context}
        )
        monitor.end_timer("save_assistant_message")
        logger.info(f"Saved assistant message {assistant_message.id} to chat {chat.id}")
        
        # 10. Generate and save chat name for new chats
        monitor.start_timer("generate_chat_name")
        if not chat.name:
            try:
                chat_name = await llm_service.generate_chat_name(request.query, full_response)
                await chat_service.update_chat_name(chat.id, chat_name)
                chat.name = chat_name  # Update local object
                logger.info(f"Generated and saved chat name for {chat.id}: {chat_name}")
            except Exception as e:
                logger.error(f"Failed to generate chat name for {chat.id}: {e}")
        monitor.end_timer("generate_chat_name")
        
        # Calculate total time
        total_time = monitor.end_timer("total_processing")
        
        # Log performance summary
        monitor.log_summary()
        
        # Identify bottlenecks
        bottlenecks = monitor.identify_bottlenecks(threshold_percentage=40.0)
        if bottlenecks:
            logger.warning(f"Performance bottlenecks detected: {bottlenecks}")
        
        return QueryResponse(
            text_response=full_response,
            audio_base64=audio_base64,
            audio_format="mp3" if audio_base64 else None,
            context_chunks=len(context),
            processing_time=total_time,
            chat_id=chat.id,
            performance_metrics=monitor.get_metrics()  # Include metrics in response
        )
        
    except Exception as e:
        logger.error(f"Error in optimized voice query processing: {e}")
        raise HTTPException(status_code=500, detail=f"Voice query failed: {str(e)}")


# Additional optimization tips implemented above:
# 1. Start vector search immediately as async task
# 2. Use streaming metrics to track performance
# 3. Choose between streaming and regular TTS based on text length
# 4. Use maximum optimization settings for ElevenLabs
# 5. Include detailed performance metrics in response
# 6. Save user and assistant messages to database
# 7. Include chat history for better context
# 8. Generate and save chat names for new chats

# Further optimizations you can implement:
# 1. Cache frequently used voice outputs
# 2. Pre-warm ElevenLabs connection
# 3. Use connection pooling for API calls
# 4. Implement request queuing and batching
# 5. Use CDN for audio delivery
# 6. Compress audio before base64 encoding 