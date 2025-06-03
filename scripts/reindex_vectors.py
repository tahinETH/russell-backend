#!/usr/bin/env python3
"""
Script to re-index vectors with source IDs instead of full content
"""
import sys
import os
import asyncio
import json

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vector import VectorService
from app.config import settings

async def main():
    """Main function to re-index vectors"""
    print("Starting vector re-indexing...")
    
    # Initialize vector service
    vector_service = VectorService(
        api_key=settings.pinecone_api_key,
        environment=settings.pinecone_environment,
        index_name=settings.pinecone_index_name,
        embedding_model=settings.embedding_model
    )
    
    # Get index stats before
    print("Current index stats:")
    stats = vector_service.get_index_stats()
    print(f"Total vectors: {stats.get('total_vector_count', 0)}")
    
    # Load documents from JSON files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    faq_file = os.path.join(script_dir, "faq.json")
    papers_file = os.path.join(script_dir, "scientific_papers.json")
    
    all_documents = []
    
    # Load FAQ data
    if os.path.exists(faq_file):
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
        print(f"Loaded {len(faq_data)} FAQ documents")
    
    # Load scientific papers data
    if os.path.exists(papers_file):
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
        print(f"Loaded {len(papers_data)} scientific paper documents")
    
    if not all_documents:
        print("No documents to index")
        return False
    
    print(f"Total documents to index: {len(all_documents)}")
    
    # Clear existing vectors (optional)
    print("Note: This will update existing vectors with new metadata structure")
    
    # Upsert documents with new structure
    print("Upserting documents with new structure...")
    success = await vector_service.upsert_documents(all_documents)
    
    if success:
        print("Vector re-indexing completed successfully!")
        
        # Get index stats after
        print("Updated index stats:")
        stats = vector_service.get_index_stats()
        print(f"Total vectors: {stats.get('total_vector_count', 0)}")
        
        return True
    else:
        print("Vector re-indexing failed!")
        return False

if __name__ == "__main__":
    # Run the main function
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 