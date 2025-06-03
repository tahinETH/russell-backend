#!/usr/bin/env python3
"""
Test script to verify the AI backend setup
"""
import sys
import os
import asyncio
import importlib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    required_modules = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'sqlalchemy',
        'asyncpg',
        'litellm',
        'pinecone',
        'sse_starlette'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module.replace('-', '_'))
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All imports successful!")
    return True

def test_app_modules():
    """Test that app modules can be imported"""
    print("\n🔍 Testing app modules...")
    
    try:
        from app.config import settings
        print("  ✅ config.py")
        
        from app.models import User, Chat, Message
        print("  ✅ models.py")
        
        from app.database import engine, Base
        print("  ✅ database.py")
        
        from app.services.llm import LLMService
        print("  ✅ services/llm.py")
        
        from app.services.vector import VectorService
        print("  ✅ services/vector.py")
        
        from app.utils.embeddings import chunk_text
        print("  ✅ utils/embeddings.py")
        
        from app.api import router
        print("  ✅ api.py")
        
        from app.main import app
        print("  ✅ main.py")
        
        print("✅ All app modules loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading app modules: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    print("\n🔍 Testing environment...")
    
    try:
        from app.config import settings
        
        # Check if .env file exists
        env_file = ".env"
        if os.path.exists(env_file):
            print(f"  ✅ {env_file} exists")
        else:
            print(f"  ⚠️  {env_file} not found (using defaults)")
        
        # Test configuration loading
        print(f"  📊 Database URL: {settings.database_url[:50]}...")
        print(f"  🤖 LLM Model: {settings.llm_model}")
        print(f"  🔍 Embedding Model: {settings.embedding_model}")
        
    
       
       
            
        if hasattr(settings, 'pinecone_api_key') and settings.pinecone_api_key:
            print("  ✅ Pinecone API key configured")
        else:
            print("  ⚠️  Pinecone API key not configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_text_processing():
    """Test text processing utilities"""
    print("\n🔍 Testing text processing...")
    
    try:
        from app.utils.embeddings import chunk_text, prepare_documents_for_indexing
        
        # Test chunking
        test_text = "This is a test document. " * 100
        chunks = chunk_text(test_text, chunk_size=50, overlap=10)
        print(f"  ✅ Text chunking: {len(chunks)} chunks created")
        
        # Test document preparation
        test_docs = [
            {"id": "test_1", "title": "Test Doc", "content": test_text}
        ]
        prepared = prepare_documents_for_indexing(test_docs)
        print(f"  ✅ Document preparation: {len(prepared)} chunks prepared")
        
        return True
        
    except Exception as e:
        print(f"❌ Text processing test failed: {e}")
        return False

async def test_database_connection():
    """Test database connection"""
    print("\n🔍 Testing database connection...")
    
    try:
        from app.database import engine
        
        # Try to connect
        async with engine.begin() as conn:
            result = await conn.execute("SELECT 1")
            print("  ✅ Database connection successful")
            return True
            
    except Exception as e:
        print(f"  ⚠️  Database connection failed: {e}")
        print("  💡 Make sure PostgreSQL is running and DATABASE_URL is correct")
        return False

def test_fastapi_app():
    """Test FastAPI app creation"""
    print("\n🔍 Testing FastAPI app...")
    
    try:
        from app.main import app
        
        print(f"  ✅ App title: {app.title}")
        print(f"  ✅ App version: {app.version}")
        print(f"  ✅ Routes: {len(app.routes)} routes configured")
        
        return True
        
    except Exception as e:
        print(f"❌ FastAPI app test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 AI Backend MVP - Setup Test\n")
    
    tests = [
        ("Package Imports", test_imports, False),
        ("App Modules", test_app_modules, False), 
        ("Environment", test_environment, False),
        ("Text Processing", test_text_processing, False),
        ("Database Connection", test_database_connection, True),
        ("FastAPI App", test_fastapi_app, False),
    ]
    
    results = []
    
    for test_name, test_func, is_async in tests:
        print(f"\n{'='*50}")
        print(f"{test_name}")
        print('='*50)
        
        try:
            if is_async:
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Create a .env file with your API keys")
        print("2. Run: python scripts/init_db.py")
        print("3. Run: python scripts/load_documents.py")
        print("4. Run: uvicorn app.main:app --reload")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    asyncio.run(main()) 