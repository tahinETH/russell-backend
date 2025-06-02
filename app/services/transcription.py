import litellm
import tempfile
import os
from typing import BinaryIO
import logging
import asyncio

logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(self, model: str = "openai/whisper-1"):
        self.model = model
    
    async def transcribe_audio(self, audio_file: BinaryIO, filename: str) -> str:
        """
        Transcribe audio file using OpenAI Whisper model via litellm
        
        Args:
            audio_file: Binary audio file content
            filename: Original filename to preserve extension
            
        Returns:
            str: Transcribed text
        """
        max_retries = 2  # Original attempt + 1 retry
        
        for attempt in range(max_retries):
            try:
                # Create a temporary file to store the audio content
                # We need to preserve the file extension for proper processing
                file_extension = os.path.splitext(filename)[1] if filename else '.wav'
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    # Write the uploaded file content to temp file
                    content = audio_file.read()
                    temp_file.write(content)
                    temp_file.flush()
                    
                    try:
                        
                        response = await litellm.atranscription(
                            model=self.model,
                            file=open(temp_file.name, 'rb')
                        )
                        
                        
                        return response.text
                        
                    finally:
                        # Clean up the temporary file
                        os.unlink(temp_file.name)
                        
            except Exception as e:
                logger.error(f"Transcription attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:  # Last attempt failed
                    raise Exception(f"Transcription failed after {max_retries} attempts: {str(e)}")
                
                # Wait a bit before retrying
                await asyncio.sleep(1)
                
                # Reset file pointer for retry
                audio_file.seek(0) 