from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import List, Optional
import uuid
import json
import asyncio
import logging

from .models import (
    QueryRequest, UserCreate, UserResponse, ChatResponse, 
    MessageResponse, ChatWithMessages,
    User, Chat, Message
)
from .database import get_db
from .services.llm import LLMService
from .services.vector import VectorService

logger = logging.getLogger(__name__)

router = APIRouter()

# Global service instances (will be injected by main.py)
llm_service: Optional[LLMService] = None
vector_service: Optional[VectorService] = None

def set_services(llm: LLMService, vector: VectorService):
    """Set the service instances"""
    global llm_service, vector_service
    llm_service = llm
    vector_service = vector

@router.post("/users", response_model=UserResponse)
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    """Create a new user"""
    try:
        # Check if username already exists
        result = await db.execute(select(User).where(User.username == user.username))
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Create new user
        new_user = User(username=user.username)
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        return UserResponse(
            id=new_user.id,
            username=new_user.username,
            created_at=new_user.created_at
        )
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create user")

@router.post("/query")
async def query_endpoint(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    """Handle streaming Q&A with vector search context"""
    if not llm_service or not vector_service:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    try:
        # 1. Verify user exists
        result = await db.execute(select(User).where(User.id == request.user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # 2. Get or create chat
        if request.chat_id:
            result = await db.execute(
                select(Chat).where(
                    Chat.id == request.chat_id,
                    Chat.user_id == request.user_id
                )
            )
            chat = result.scalar_one_or_none()
            if not chat:
                raise HTTPException(status_code=404, detail="Chat not found")
        else:
            chat = Chat(user_id=request.user_id)
            db.add(chat)
            await db.commit()
            await db.refresh(chat)
        
        # 3. Save user message
        user_message = Message(
            chat_id=chat.id,
            role="user",
            content=request.query
        )
        db.add(user_message)
        await db.commit()
        
        # 4. Get context from Pinecone
        context = await vector_service.search(request.query)
        
        # 5. Stream response
        async def generate():
            full_response = ""
            try:
                yield f"data: {json.dumps({'type': 'start', 'chat_id': str(chat.id)})}\n\n"
                
                async for chunk in llm_service.stream_with_context(request.query, context):
                    full_response += chunk
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                
                # Save assistant message
                assistant_message = Message(
                    chat_id=chat.id,
                    role="assistant",
                    content=full_response,
                    context={"retrieved_chunks": context}
                )
                db.add(assistant_message)
                await db.commit()
                
                yield f"data: {json.dumps({'type': 'end', 'content': full_response})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        
        return EventSourceResponse(generate())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/users/{user_id}/chats", response_model=List[ChatResponse])
async def get_user_chats(user_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Get all chats for a user"""
    try:
        result = await db.execute(
            select(Chat)
            .where(Chat.user_id == user_id)
            .order_by(Chat.created_at.desc())
        )
        chats = result.scalars().all()
        
        return [
            ChatResponse(
                id=chat.id,
                user_id=chat.user_id,
                created_at=chat.created_at
            )
            for chat in chats
        ]
    except Exception as e:
        logger.error(f"Error getting user chats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chats")

@router.get("/chats/{chat_id}/messages", response_model=List[MessageResponse])
async def get_chat_messages(chat_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Get all messages for a chat"""
    try:
        result = await db.execute(
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at.asc())
        )
        messages = result.scalars().all()
        
        return [
            MessageResponse(
                id=msg.id,
                chat_id=msg.chat_id,
                role=msg.role,
                content=msg.content,
                context=msg.context,
                created_at=msg.created_at
            )
            for msg in messages
        ]
    except Exception as e:
        logger.error(f"Error getting chat messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve messages")

@router.get("/chats/{chat_id}", response_model=ChatWithMessages)
async def get_chat_with_messages(chat_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Get a chat with all its messages"""
    try:
        result = await db.execute(
            select(Chat)
            .options(selectinload(Chat.messages))
            .where(Chat.id == chat_id)
        )
        chat = result.scalar_one_or_none()
        
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        messages = [
            MessageResponse(
                id=msg.id,
                chat_id=msg.chat_id,
                role=msg.role,
                content=msg.content,
                context=msg.context,
                created_at=msg.created_at
            )
            for msg in sorted(chat.messages, key=lambda x: x.created_at)
        ]
        
        return ChatWithMessages(
            id=chat.id,
            user_id=chat.user_id,
            created_at=chat.created_at,
            messages=messages
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat with messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat")

@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Delete a chat and all its messages"""
    try:
        # Delete messages first
        await db.execute(
            Message.__table__.delete().where(Message.chat_id == chat_id)
        )
        
        # Delete chat
        result = await db.execute(
            Chat.__table__.delete().where(Chat.id == chat_id)
        )
        
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        await db.commit()
        return {"message": "Chat deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete chat")

@router.get("/vector/stats")
async def get_vector_stats():
    """Get Pinecone index statistics"""
    if not vector_service:
        raise HTTPException(status_code=500, detail="Vector service not initialized")
    
    try:
        stats = vector_service.get_index_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting vector stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get vector stats") 