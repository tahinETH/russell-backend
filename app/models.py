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
    images = relationship("MessageImage", back_populates="message")

class MessageImage(Base):
    __tablename__ = "message_images"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"))
    prompt = Column(Text, nullable=False)  # The image generation prompt
    image_url = Column(String, nullable=False)  # The generated image URL
    created_at = Column(DateTime, default=datetime.utcnow)
    
    message = relationship("Message", back_populates="images")

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
    lesson: Optional[str] = None  # Add lesson parameter for specific experiences like "blackholes"
    expertise: Optional[int] = 3  # Expertise level (1-5, default 3)

class QueryResponse(BaseModel):
    text_response: str
    audio_base64: Optional[str] = None
    audio_format: Optional[str] = None
    context_chunks: int = 0
    processing_time: float
    chat_id: uuid_module.UUID
    lesson: Optional[str] = None  # Return the lesson parameter in response

class UserCreate(BaseModel):
    username: Optional[str] = None
    email: str
    name: Optional[str] = None
    fe_metadata: Optional[dict] = None

class UserResponse(BaseModel):
    id: str
    username: Optional[str] = None
    email: str
    name: Optional[str] = None
    fe_metadata: Optional[dict] = None
    created_at: datetime

class MessageImageResponse(BaseModel):
    id: uuid_module.UUID
    message_id: uuid_module.UUID
    prompt: str
    image_url: str
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
    images: List[MessageImageResponse] = []

class ChatWithMessages(BaseModel):
    id: uuid_module.UUID
    user_id: str  # Changed to string
    name: Optional[str] = None  # Auto-generated chat name
    created_at: datetime
    messages: List[MessageResponse]

class TranscriptionResponse(BaseModel):
    transcription: str
    message: str = "Audio transcribed successfully"

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