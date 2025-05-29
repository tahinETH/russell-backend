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
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    chats = relationship("Chat", back_populates="user")

class Chat(Base):
    __tablename__ = "chats"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
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

# Pydantic Models
class QueryRequest(BaseModel):
    query: str
    user_id: uuid_module.UUID
    chat_id: Optional[uuid_module.UUID] = None

class UserCreate(BaseModel):
    username: str

class UserResponse(BaseModel):
    id: uuid_module.UUID
    username: str
    created_at: datetime

class ChatResponse(BaseModel):
    id: uuid_module.UUID
    user_id: uuid_module.UUID
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
    user_id: uuid_module.UUID
    created_at: datetime
    messages: List[MessageResponse] 