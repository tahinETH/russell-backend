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
    
    def __init__(self, use_streaming: bool = False):
        self.api_key = settings.elevenlabs_api_key
        self.voice_id = settings.elevenlabs_voice_id
        self.model_id = settings.elevenlabs_model_id
        self.use_streaming = use_streaming  # Flag to switch between streaming and HTTP
        
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
        """
        Convert text to speech using HTTP request (non-streaming)
        
        Args:
            text: Text to convert to speech
            voice_id: Optional voice ID override
            model_id: Optional model ID override
            output_format: Audio output format
            
        Returns:
            Complete audio data as bytes
        """
        if not self.client:
            logger.error("ElevenLabs API key not configured")
            raise ValueError("ElevenLabs service not configured")
        
        voice_id = voice_id or self.voice_id
        model_id = model_id or self.model_id
        voice_settings = VoiceSettings(
            stability=0.75,
            similarity_boost=1,
            speed=0.9,
            style=0.6,
            use_speaker_boost=True
        )

        try:
            # Use the convert method for HTTP request
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
    
    # STREAMING METHODS - KEPT FOR FUTURE USE
    # These methods are preserved in case you want to switch back to streaming
    
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
        
        DEPRECATED: This method is kept for future use if streaming is needed again.
        Use text_to_speech() for HTTP requests instead.
        
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
            # Use the stream() method as per the official API
            audio_stream = self.client.text_to_speech.stream(
                voice_id=voice_id,
                output_format=output_format,
                text=text,
                model_id=model_id,
                voice_settings=voice_settings
            )
            
            # Stream the chunks directly
            for chunk in audio_stream:
                if chunk:
                    yield chunk
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.001)
                    
        except Exception as e:
            logger.error(f"Error in text-to-speech streaming: {e}")
            raise
    
    def enable_streaming(self):
        """Enable streaming mode for future requests"""
        self.use_streaming = True
        logger.info("ElevenLabs service switched to streaming mode")
    
    def disable_streaming(self):
        """Disable streaming mode, use HTTP requests"""
        self.use_streaming = False
        logger.info("ElevenLabs service switched to HTTP request mode")
    
    def encode_audio_base64(self, audio_bytes: bytes) -> str:
        """Encode audio bytes to base64 string"""
        return base64.b64encode(audio_bytes).decode('utf-8') 