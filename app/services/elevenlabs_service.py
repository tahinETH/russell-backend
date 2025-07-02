import os
import logging
import base64
from typing import Optional, AsyncGenerator
from elevenlabs import ElevenLabs, VoiceSettings
import asyncio
from ..config import settings

logger = logging.getLogger(__name__)

class ElevenLabsService:
    """Service for text-to-speech using ElevenLabs API"""
    
    def __init__(self):
        self.api_key = settings.elevenlabs_api_key
        self.voice_id = settings.elevenlabs_voice_id
        self.model_id = settings.elevenlabs_model_id
        
        if not self.api_key:
            logger.warning("ElevenLabs API key not found. Voice synthesis will be disabled.")
            self.client = None
        else:
            # Initialize sync client only
            self.client = ElevenLabs(api_key=self.api_key)
    
    async def text_to_speech(
        self, 
        text: str,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        output_format: str = "mp3_44100_128"
    ) -> bytes:
        if not self.client:
            logger.error("ElevenLabs API key not configured")
            raise ValueError("ElevenLabs service not configured")
        
        voice_id = voice_id or self.voice_id
        model_id = model_id or self.model_id
        voice_settings = VoiceSettings(
            stability=0.8,
            similarity_boost=1,
            speed=1,
            use_speaker_boost=True
        )

        try:
            # Use the convert method as specified
            audio_data = self.client.text_to_speech.convert(
                voice_id=voice_id,
                output_format=output_format,
                text=text,
                model_id=model_id,
                voice_settings=voice_settings
            )
            
            # Convert generator to bytes if needed
            if hasattr(audio_data, '__iter__') and not isinstance(audio_data, (bytes, str)):
                # If it's a generator/iterator, collect all chunks
                audio_chunks = []
                for chunk in audio_data:
                    audio_chunks.append(chunk)
                return b''.join(audio_chunks)
            else:
                # If it's already bytes, return directly
                return audio_data
                    
        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {e}")
            raise
    
    async def text_to_speech_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        output_format: str = "mp3_44100_128",
        chunk_size: int = 1024,
        optimize_streaming_latency: int = 3,
        stream_chunk_size: int = 2048
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text to speech conversion using ElevenLabs API with optimizations
        
        Args:
            text: Text to convert to speech
            voice_id: Optional voice ID override
            model_id: Optional model ID override
            output_format: Audio output format
            chunk_size: Size of audio chunks to yield
            optimize_streaming_latency: Optimization level (0-4, higher = lower latency)
            stream_chunk_size: Size of chunks for streaming
            
        Yields:
            Audio data chunks as bytes
        """
        if not self.client:
            logger.error("ElevenLabs API key not configured")
            raise ValueError("ElevenLabs service not configured")
        
        voice_id = voice_id or self.voice_id
        model_id = model_id or self.model_id
        voice_settings = VoiceSettings(
            stability=0.8,
            similarity_boost=1,
            speed=0.8,
            use_speaker_boost=True
        )
        
        
        try:
            # Try to use the streaming endpoint if available
            # Check if the client has a streaming method
            if hasattr(self.client.text_to_speech, 'convert_as_stream'):
                # Use the streaming endpoint
                audio_stream = self.client.text_to_speech.convert_as_stream(
                    voice_id=voice_id,
                    output_format=output_format,
                    text=text,
                    model_id=model_id,
                    optimize_streaming_latency=optimize_streaming_latency,
                    stream_chunk_size=stream_chunk_size,
                    voice_settings=voice_settings
                )
                
                # Stream the chunks directly
                for chunk in audio_stream:
                    if chunk:
                        yield chunk
            else:
                # Fallback to the regular method with manual chunking
                # Run the sync method in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                
                audio_stream = await loop.run_in_executor(
                    None,
                    lambda: self.client.text_to_speech.convert(
                        voice_id=voice_id,
                        output_format=output_format,
                        text=text,
                        model_id=model_id,
                        stream=True  # Enable streaming if supported
                    )
                )
                
                # If we get a generator/iterator, stream the chunks
                if hasattr(audio_stream, '__iter__') and not isinstance(audio_stream, (bytes, str)):
                    for chunk in audio_stream:
                        if chunk:
                            yield chunk
                            # Small delay to prevent overwhelming the client
                            await asyncio.sleep(0.001)
                else:
                    # If we get bytes directly, chunk them manually
                    if isinstance(audio_stream, bytes):
                        for i in range(0, len(audio_stream), chunk_size):
                            yield audio_stream[i:i + chunk_size]
                            await asyncio.sleep(0.001)
        except Exception as e:
            logger.error(f"Error in text-to-speech streaming: {e}")
            raise
    
    def encode_audio_base64(self, audio_bytes: bytes) -> str:
        """Encode audio bytes to base64 string"""
        return base64.b64encode(audio_bytes).decode('utf-8') 