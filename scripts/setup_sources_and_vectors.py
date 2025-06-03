#!/usr/bin/env python3
"""
Comprehensive setup script to:
1. Create the sources table
2. Populate it with data from JSON files  
3. Re-index vectors with new source ID approach
"""
import sys
import os
import asyncio
import json

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import engine, Base
from app.models import Source
from app.services.source_service import SourceService
from app.services.vector import VectorService
from app.config import settings

async def create_sources_table():
    """Create the sources table"""
    print("1. Creating sources table...")
    
    try:
        # Create all tables (this will only create new ones)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        print("✓ Sources table created successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error creating sources table: {e}")
        return False

async def populate_sources():
    """Populate sources table from JSON files"""
    print("\n2. Populating sources table...")
    
    # Initialize source service
    source_service = SourceService()
    
    # Get paths to JSON files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    faq_file = os.path.join(script_dir, "faq.json")
    papers_file = os.path.join(script_dir, "scientific_papers.json")
    
    # Check if files exist
    if not os.path.exists(faq_file):
        print(f"✗ Error: FAQ file not found at {faq_file}")
        return False
    
    if not os.path.exists(papers_file):
        print(f"✗ Error: Scientific papers file not found at {papers_file}")
        return False
    
    # Delete existing sources (for clean re-population)
    print("   Deleting existing sources...")
    await source_service.delete_all_sources()
    
    # Populate sources from JSON files
    print("   Loading data from JSON files...")
    success = await source_service.populate_from_json_files(faq_file, papers_file)
    
    if success:
        # Get count of sources
        all_sources = await source_service.get_all_sources()
        faq_count = len([s for s in all_sources if s.source_type == "faq"])
        papers_count = len([s for s in all_sources if s.source_type == "scientific_papers"])
        
        print(f"✓ Sources populated successfully!")
        print(f"   FAQ sources: {faq_count}")
        print(f"   Scientific papers: {papers_count}")
        print(f"   Total sources: {len(all_sources)}")
        return True
    else:
        print("✗ Sources population failed!")
        return False

async def reindex_vectors():
    """Re-index vectors with new source ID approach"""
    print("\n3. Re-indexing vectors...")
    
    # Initialize vector service
    vector_service = VectorService(
        api_key=settings.pinecone_api_key,
        environment=settings.pinecone_environment,
        index_name=settings.pinecone_index_name,
        embedding_model=settings.embedding_model
    )
    
    # Get index stats before
    print("   Current index stats:")
    stats = vector_service.get_index_stats()
    print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
    
    # Load documents from JSON files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    faq_file = os.path.join(script_dir, "faq.json")
    papers_file = os.path.join(script_dir, "scientific_papers.json")
    
    all_documents = []
    
    # Load FAQ data
    with open(faq_file, 'r', encoding='utf-8') as f:
        faq_data = json.load(f)
    for item in faq_data:
        all_documents.append({
            "id": item["id"],
            "title": item["title"],
            "content": item["content"],
            "link": item.get("link", ""),
            "source_type": "faq"
        })
    
    # Load scientific papers data
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers_data = json.load(f)
    for item in papers_data:
        all_documents.append({
            "id": item["id"],
            "title": item["title"],
            "content": item["content"],
            "link": item.get("link", ""),
            "source_type": "scientific_papers"
        })
    
    print(f"   Documents to index: {len(all_documents)}")
    
    # Upsert documents with new structure
    print("   Upserting vectors with new metadata structure...")
    success = await vector_service.upsert_documents(all_documents)
    
    if success:
        # Get index stats after
        print("   Updated index stats:")
        stats = vector_service.get_index_stats()
        print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
        print("✓ Vector re-indexing completed successfully!")
        return True
    else:
        print("✗ Vector re-indexing failed!")
        return False

async def test_search():
    """Test the new search functionality"""
    print("\n4. Testing search functionality...")
    
    # Initialize vector service
    vector_service = VectorService(
        api_key=settings.pinecone_api_key,
        environment=settings.pinecone_environment,
        index_name=settings.pinecone_index_name,
        embedding_model=settings.embedding_model
    )
    
    # Test search
    test_query = "What is loomlock?"
    print(f"   Testing query: '{test_query}'")
    
    try:
        results = await vector_service.search(test_query, top_k=3)
        
        if results:
            print(f"✓ Search successful! Found {len(results)} results")
            for i, result in enumerate(results[:2], 1):
                print(f"   Result {i}:")
                print(f"     Score: {result['score']:.4f}")
                print(f"     Title: {result['metadata']['title'][:100]}...")
                print(f"     Source Type: {result['metadata']['source_type']}")
                print(f"     Content Length: {len(result['content'])} characters")
                print(f"     Content Preview: {result['content'][:200]}...")
        else:
            print("✗ Search returned no results")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Search test failed: {e}")
        return False

async def main():
    """Main setup function"""
    print("=== Setting up Sources Table and Vector Search ===\n")
    
    # Step 1: Create sources table
    success1 = await create_sources_table()
    if not success1:
        print("\nSetup failed at table creation step.")
        return False
    
    # Step 2: Populate sources
    success2 = await populate_sources()
    if not success2:
        print("\nSetup failed at sources population step.")
        return False
    
    # Step 3: Re-index vectors
    success3 = await reindex_vectors()
    if not success3:
        print("\nSetup failed at vector re-indexing step.")
        return False
    
    # Step 4: Test search
    success4 = await test_search()
    if not success4:
        print("\nSetup failed at search testing step.")
        return False
    
    print("\n=== Setup completed successfully! ===")
    print("The system is now configured to:")
    print("- Store full content in the sources database table")
    print("- Use vector search to find relevant source IDs")
    print("- Fetch full content from database using source IDs")
    print("- Provide complete, untruncated content to the LLM")
    
    return True

if __name__ == "__main__":
    # Run the main function
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 