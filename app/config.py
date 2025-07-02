from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    # Database - SQLite for local development
    database_url: str = "sqlite+aiosqlite:///./aiapp.db"
    
    # LiteLLM
    #llm_model: str = "claude-3-5-sonnet-latest"
    #llm_model: str = "claude-3-opus-20240229"
    llm_model: str = "claude-opus-4-20250514"
    
    # Pinecone - using your variable names
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME")
    embedding_model: str = "text-embedding-3-large"
    embedding_ctx_length: int = 8191
    chunk_size: int = 600
    
    # Clerk Configuration
    CLERK_WEBHOOK_SECRET: str = os.getenv("CLERK_WEBHOOK_SECRET", "")
    CLERK_SECRET_KEY: str = os.getenv("CLERK_SECRET_KEY", "")
    
    # Stripe Configuration
    
    # ElevenLabs Configuration
    elevenlabs_api_key: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "NFG5qt843uXKj4pFvR7C")
    elevenlabs_model_id: str = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # This allows extra env vars without errors

settings = Settings()

# Create a config object for backward compatibility
class Config:
    CLERK_WEBHOOK_SECRET = settings.CLERK_WEBHOOK_SECRET
    CLERK_SECRET_KEY = settings.CLERK_SECRET_KEY

config = Config()