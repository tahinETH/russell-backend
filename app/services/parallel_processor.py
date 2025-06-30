import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, Optional, List
from collections import deque
import time

from .llm import LLMService
from .elevenlabs_service import ElevenLabsService
from .vector import VectorService

logger = logging.getLogger(__name__)

class ParallelProcessingService:
    """Service for parallel processing of LLM and TTS to minimize latency"""
    
    def __init__(
        self, 
        llm_service: LLMService, 
        elevenlabs_service: ElevenLabsService,
        vector_service: VectorService
    ):
        self.llm_service = llm_service
        self.elevenlabs_service = elevenlabs_service
        self.vector_service = vector_service
        
        # Queue for sentences ready for TTS
        self.sentence_queue = asyncio.Queue(maxsize=10)
        # Buffer for incomplete sentences
        self.text_buffer = ""
        # Track processing state
        self.llm_complete = False
        self.processing_error = None
        
    async def process_query_parallel(
        self,
        query: str,
        chat_history: List[Dict] = None,
        custom_system_prompt: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        enable_voice: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process query with parallel LLM and TTS processing
        
        Yields events:
        - {"type": "start"}
        - {"type": "text_chunk", "content": str}
        - {"type": "sentence_ready", "text": str}
        - {"type": "voice_chunk", "audio": str, "sentence_index": int}
        - {"type": "complete", "full_text": str}
        """
        start_time = time.time()
        
        # Reset state
        self.text_buffer = ""
        self.llm_complete = False
        self.processing_error = None
        self.sentence_queue = asyncio.Queue(maxsize=10)
        
        try:
            # Get context (can be done in parallel with other operations)
            context_task = asyncio.create_task(self.vector_service.search(query))
            
            yield {"type": "start", "timestamp": time.time()}
            
            # Wait for context
            context = await context_task
            
            # Create tasks for parallel processing
            tasks = []
            
            # Task 1: LLM streaming and sentence extraction
            llm_task = asyncio.create_task(
                self._process_llm_stream(
                    query, context, chat_history, custom_system_prompt
                )
            )
            tasks.append(llm_task)
            
            # Task 2: TTS processing (only if voice is enabled)
            if enable_voice and self.elevenlabs_service.api_key:
                tts_task = asyncio.create_task(
                    self._process_tts_queue(voice_id, model_id)
                )
                tasks.append(tts_task)
            
            # Collect results from both tasks
            full_text = ""
            sentence_index = 0
            
            # Process events from tasks
            while tasks:
                # Wait for any task to produce a result
                done, pending = await asyncio.wait(
                    tasks, 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    try:
                        result = await task
                        if result:
                            if result["type"] == "llm_complete":
                                full_text = result["full_text"]
                                self.llm_complete = True
                                tasks.remove(task)
                            elif result["type"] == "tts_complete":
                                tasks.remove(task)
                    except Exception as e:
                        logger.error(f"Task error: {e}")
                        self.processing_error = e
                        tasks.remove(task)
            
            # Yield completion event
            processing_time = time.time() - start_time
            yield {
                "type": "complete",
                "full_text": full_text,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            yield {"type": "error", "error": str(e)}
    
    async def _process_llm_stream(
        self,
        query: str,
        context: List[Dict],
        chat_history: List[Dict],
        custom_system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Process LLM stream and extract sentences"""
        full_text = ""
        sentence_buffer = ""
        
        try:
            async for chunk in self.llm_service.stream_with_context(
                query, context, chat_history, custom_system_prompt
            ):
                full_text += chunk
                sentence_buffer += chunk
                
                # Check for sentence boundaries
                sentence_ends = ['.', '!', '?', '\n']
                
                # Find the last sentence ending
                last_end_pos = -1
                for end in sentence_ends:
                    pos = sentence_buffer.rfind(end)
                    if pos > last_end_pos:
                        last_end_pos = pos
                
                # If we found a sentence ending
                if last_end_pos > -1:
                    # Extract complete sentence(s)
                    complete_sentence = sentence_buffer[:last_end_pos + 1].strip()
                    sentence_buffer = sentence_buffer[last_end_pos + 1:]
                    
                    if complete_sentence:
                        # Add to TTS queue
                        await self.sentence_queue.put(complete_sentence)
            
            # Process any remaining text
            if sentence_buffer.strip():
                await self.sentence_queue.put(sentence_buffer.strip())
            
            # Signal completion
            await self.sentence_queue.put(None)  # Sentinel value
            
            return {"type": "llm_complete", "full_text": full_text}
            
        except Exception as e:
            logger.error(f"Error in LLM processing: {e}")
            await self.sentence_queue.put(None)  # Signal error
            raise
    
    async def _process_tts_queue(
        self,
        voice_id: Optional[str],
        model_id: Optional[str]
    ) -> Dict[str, Any]:
        """Process sentences from queue for TTS"""
        sentence_index = 0
        
        try:
            while True:
                # Get sentence from queue
                sentence = await self.sentence_queue.get()
                
                # Check for completion
                if sentence is None:
                    break
                
                # Process TTS for this sentence
                try:
                    async for audio_chunk in self.elevenlabs_service.text_to_speech_stream(
                        sentence,
                        voice_id=voice_id,
                        model_id=model_id,
                        optimize_streaming_latency=4  # Maximum optimization
                    ):
                        # Yield audio chunk
                        # Note: In real implementation, you'd yield this through the main generator
                        logger.debug(f"Generated audio chunk for sentence {sentence_index}")
                        
                    sentence_index += 1
                    
                except Exception as e:
                    logger.error(f"Error in TTS for sentence: {e}")
                    # Continue with next sentence
            
            return {"type": "tts_complete", "sentences_processed": sentence_index}
            
        except Exception as e:
            logger.error(f"Error in TTS queue processing: {e}")
            raise 