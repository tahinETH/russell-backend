from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from typing import List, Dict, Optional
from ..models import Source, SourceResponse
from ..database import AsyncSessionLocal
import logging
import json
import os

logger = logging.getLogger(__name__)

class SourceService:
    """Service for managing sources in the database"""
    
    async def get_source_by_id(self, source_id: str) -> Optional[Source]:
        """Get a source by its ID"""
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(Source).where(Source.id == source_id)
                )
                return result.scalar_one_or_none()
            except Exception as e:
                logger.error(f"Error getting source {source_id}: {e}")
                return None
    
    async def get_sources_by_ids(self, source_ids: List[str]) -> List[Source]:
        """Get multiple sources by their IDs"""
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(Source).where(Source.id.in_(source_ids))
                )
                return result.scalars().all()
            except Exception as e:
                logger.error(f"Error getting sources {source_ids}: {e}")
                return []
    
    async def create_source(self, source_data: Dict) -> Optional[Source]:
        """Create a new source"""
        async with AsyncSessionLocal() as session:
            try:
                source = Source(
                    id=source_data["id"],
                    title=source_data["title"],
                    content=source_data["content"],
                    link=source_data.get("link", ""),
                    source_type=source_data["source_type"]
                )
                session.add(source)
                await session.commit()
                await session.refresh(source)
                return source
            except Exception as e:
                logger.error(f"Error creating source: {e}")
                await session.rollback()
                return None
    
    async def bulk_create_sources(self, sources_data: List[Dict]) -> bool:
        """Create multiple sources in bulk"""
        async with AsyncSessionLocal() as session:
            try:
                sources = []
                for source_data in sources_data:
                    source = Source(
                        id=source_data["id"],
                        title=source_data["title"],
                        content=source_data["content"],
                        link=source_data.get("link", ""),
                        source_type=source_data["source_type"]
                    )
                    sources.append(source)
                
                session.add_all(sources)
                await session.commit()
                logger.info(f"Successfully created {len(sources)} sources")
                return True
            except Exception as e:
                logger.error(f"Error bulk creating sources: {e}")
                await session.rollback()
                return False
    
    async def populate_from_json_files(self, faq_file: str, papers_file: str) -> bool:
        """Populate sources table from JSON files"""
        try:
            # Load FAQ data
            faq_sources = []
            if os.path.exists(faq_file):
                with open(faq_file, 'r', encoding='utf-8') as f:
                    faq_data = json.load(f)
                for item in faq_data:
                    faq_sources.append({
                        "id": item["id"],
                        "title": item["title"],
                        "content": item["content"],
                        "link": item.get("link", ""),
                        "source_type": "faq"
                    })
                logger.info(f"Loaded {len(faq_sources)} FAQ sources")
            
            # Load scientific papers data
            papers_sources = []
            if os.path.exists(papers_file):
                with open(papers_file, 'r', encoding='utf-8') as f:
                    papers_data = json.load(f)
                for item in papers_data:
                    papers_sources.append({
                        "id": item["id"],
                        "title": item["title"],
                        "content": item["content"],
                        "link": item.get("link", ""),
                        "source_type": "scientific_papers"
                    })
                logger.info(f"Loaded {len(papers_sources)} scientific paper sources")
            
            # Combine all sources
            all_sources = faq_sources + papers_sources
            
            if not all_sources:
                logger.warning("No sources to populate")
                return False
            
            # Bulk create sources
            success = await self.bulk_create_sources(all_sources)
            if success:
                logger.info(f"Successfully populated {len(all_sources)} sources in the database")
            
            return success
            
        except Exception as e:
            logger.error(f"Error populating sources from JSON files: {e}")
            return False
    
    async def get_all_sources(self) -> List[Source]:
        """Get all sources"""
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(select(Source))
                return result.scalars().all()
            except Exception as e:
                logger.error(f"Error getting all sources: {e}")
                return []
    
    async def delete_all_sources(self) -> bool:
        """Delete all sources (useful for re-population)"""
        async with AsyncSessionLocal() as session:
            try:
                await session.execute(text("DELETE FROM sources"))
                await session.commit()
                logger.info("Successfully deleted all sources")
                return True
            except Exception as e:
                logger.error(f"Error deleting all sources: {e}")
                await session.rollback()
                return False 