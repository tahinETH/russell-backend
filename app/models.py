from sqlalchemy import Column, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from .database import Base
import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List
import uuid as uuid_module

# SQLAlchemy Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)  # Using Clerk user ID as primary key
    username = Column(String, unique=True, nullable=True)  # Allow null initially
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=True)
    custom_system_prompt = Column(Text, nullable=True)  # Custom system prompt for the user

    fe_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    chats = relationship("Chat", back_populates="user")

class Chat(Base):
    __tablename__ = "chats"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, ForeignKey("users.id"))  # Changed to String to match User.id
    name = Column(String, nullable=True)  # Auto-generated chat name
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id = Column(UUID(as_uuid=True), ForeignKey("chats.id"))
    role = Column(String)  # "user" or "assistant"
    content = Column(Text)
    context = Column(JSON, nullable=True)  # Store retrieved chunks
    created_at = Column(DateTime, default=datetime.utcnow)
    
    chat = relationship("Chat", back_populates="messages")

class Source(Base):
    __tablename__ = "sources"
    
    id = Column(String, primary_key=True)  # Use the original ID from JSON (e.g., "faq_001", "paper_001")
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)  # Full content
    link = Column(String, nullable=True)
    source_type = Column(String, nullable=False)  # "faq" or "scientific_papers"
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic Models
class QueryRequest(BaseModel):
    query: str
    chat_id: Optional[uuid_module.UUID] = None
    enable_voice: Optional[bool] = True
    voice_id: Optional[str] = None
    model_id: Optional[str] = None
    voice_settings: Optional[dict] = None  # ElevenLabs voice settings

class QueryResponse(BaseModel):
    text_response: str
    audio_base64: Optional[str] = None
    audio_format: Optional[str] = None
    context_chunks: int = 0
    processing_time: float
    chat_id: uuid_module.UUID

class UserCreate(BaseModel):
    username: Optional[str] = None
    email: str
    name: Optional[str] = None
    custom_system_prompt: Optional[str] = None
    fe_metadata: Optional[dict] = None

class UserResponse(BaseModel):
    id: str
    username: Optional[str] = None
    email: str
    name: Optional[str] = None
    custom_system_prompt: Optional[str] = None
    fe_metadata: Optional[dict] = None
    created_at: datetime

class ChatResponse(BaseModel):
    id: uuid_module.UUID
    user_id: str  # Changed to string
    name: Optional[str] = None  # Auto-generated chat name
    created_at: datetime

class MessageResponse(BaseModel):
    id: uuid_module.UUID
    chat_id: uuid_module.UUID
    role: str
    content: str
    context: Optional[dict] = None
    created_at: datetime

class ChatWithMessages(BaseModel):
    id: uuid_module.UUID
    user_id: str  # Changed to string
    name: Optional[str] = None  # Auto-generated chat name
    created_at: datetime
    messages: List[MessageResponse]

class TranscriptionResponse(BaseModel):
    transcription: str
    message: str = "Audio transcribed successfully"

class CustomPromptRequest(BaseModel):
    custom_system_prompt: Optional[str] = None  # None means reset to default

class CustomPromptResponse(BaseModel):
    custom_system_prompt: Optional[str] = None
    is_default: bool

class SourceResponse(BaseModel):
    id: str
    title: str
    content: str
    link: Optional[str] = None
    source_type: str
    created_at: datetime

class VoiceSettingsRequest(BaseModel):
    voice_id: Optional[str] = None
    model_id: Optional[str] = None
    stability: Optional[float] = 0.5
    similarity_boost: Optional[float] = 0.75
    style: Optional[float] = 0.0
    use_speaker_boost: Optional[bool] = True 