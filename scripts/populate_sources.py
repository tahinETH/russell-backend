#!/usr/bin/env python3
"""
Script to populate the sources table with data from faq.json and scientific_papers.json
"""
import sys
import os
import asyncio

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.source_service import SourceService

async def main():
    """Main function to populate sources"""
    print("Starting sources population...")
    
    # Initialize source service
    source_service = SourceService()
    
    # Get paths to JSON files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    faq_file = os.path.join(script_dir, "faq.json")
    papers_file = os.path.join(script_dir, "scientific_papers.json")
    
    print(f"FAQ file: {faq_file}")
    print(f"Papers file: {papers_file}")
    
    # Check if files exist
    if not os.path.exists(faq_file):
        print(f"Error: FAQ file not found at {faq_file}")
        return False
    
    if not os.path.exists(papers_file):
        print(f"Error: Scientific papers file not found at {papers_file}")
        return False
    
    # Delete existing sources (optional, for clean re-population)
    print("Deleting existing sources...")
    await source_service.delete_all_sources()
    
    # Populate sources from JSON files
    print("Populating sources from JSON files...")
    success = await source_service.populate_from_json_files(faq_file, papers_file)
    
    if success:
        print("Sources population completed successfully!")
        
        # Get count of sources
        all_sources = await source_service.get_all_sources()
        print(f"Total sources in database: {len(all_sources)}")
        
        # Show breakdown by type
        faq_count = len([s for s in all_sources if s.source_type == "faq"])
        papers_count = len([s for s in all_sources if s.source_type == "scientific_papers"])
        print(f"FAQ sources: {faq_count}")
        print(f"Scientific papers sources: {papers_count}")
        
        return True
    else:
        print("Sources population failed!")
        return False

if __name__ == "__main__":
    # Run the main function
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 