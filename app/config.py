from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    # Database - SQLite for local development
    database_url: str = "sqlite+aiosqlite:///./aiapp.db"
    
    # LiteLLM
    llm_model: str = "claude-3-5-sonnet-20240620"
    
    # Pinecone - using your variable names
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME")
    embedding_model: str = "text-embedding-3-large"
    embedding_ctx_length: int = 8191
    chunk_size: int = 600
    
    
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # This allows extra env vars without errors
    


settings = Settings()