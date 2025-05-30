from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .database import engine, Base
from .api import router, set_services
from .websocket import router as websocket_router, set_websocket_services
from .services import LLMService, VectorService, ChatService, UserService
from .config import settings
# Import models so SQLAlchemy can create tables
from .models import User, Chat, Message
# Import webhook router
from .webhooks.clerk import router as webhook_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    try:
        # Create database tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")
        
        # Initialize services
        llm_service = LLMService(settings.llm_model)
        vector_service = VectorService(
            settings.pinecone_api_key,
            settings.pinecone_environment,
            settings.pinecone_index_name
        )
        chat_service = ChatService(llm_service, vector_service)
        user_service = UserService()
        
        # Set services in API module
        set_services(llm_service, vector_service)
        # Set services in WebSocket module
        set_websocket_services(llm_service, vector_service, chat_service, user_service)
        logger.info("Services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")

app = FastAPI(
    title="AI Backend MVP",
    description="FastAPI backend with streaming Q&A, Pinecone vector search, and chat history",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)
app.include_router(websocket_router)
app.include_router(webhook_router, prefix="/webhooks")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Backend MVP",
        "version": "1.0.0",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 