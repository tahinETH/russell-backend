from pinecone import Pinecone
from litellm import embedding
from typing import List, Dict
import asyncio
import logging
from .source_service import SourceService

logger = logging.getLogger(__name__)

class VectorService:
    def __init__(self, api_key: str, environment: str, index_name: str, embedding_model: str = "text-embedding-3-large"):
        try:
            # Use new Pinecone class instead of pinecone.init()
            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index(index_name)
            self.embedding_model = embedding_model
            self.source_service = SourceService()
            logger.info(f"Initialized VectorService with index: {index_name} and model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize VectorService: {e}")
            raise
    
    async def search(self, query: str, top_k: int = 10) -> List[Dict]:
        
        try:
            # Generate embedding using LiteLLM
            query_clean = query.replace("\n", " ")
            response = embedding(model=self.embedding_model, input=[query_clean])
            query_embedding = response.data[0]['embedding']
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            
            # Extract source IDs from vector results
            source_ids = []
            vector_results = []
            
            for match in results.matches:
                source_id = match.metadata.get("source_id")
                if source_id:
                    source_ids.append(source_id)
                    vector_results.append({
                        "id": match.id,
                        "score": match.score,
                        "source_id": source_id,
                        "metadata": match.metadata
                    })
            
            
            if source_ids:
                
                sources = await self.source_service.get_sources_by_ids(source_ids)
                # Create a mapping of source_id to full source data
                source_map = {source.id: source for source in sources}
                
                
                # Combine vector results with full source content
                enriched_results = []
                for vector_result in vector_results:
                    source_id = vector_result["source_id"]
                    source = source_map.get(source_id)
                    
                    if source:
                        enriched_results.append({
                            "id": vector_result["id"],
                            "score": vector_result["score"],
                            "content": source.content,  # Full content from database
                            "metadata": {
                                "source_id": source.id,
                                "title": source.title,
                                "link": source.link,
                                "source_type": source.source_type,
                                **vector_result["metadata"]
                            }
                        })
                
                return enriched_results
            
            return []
            
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    async def upsert_documents(self, documents: List[Dict]) -> bool:
        """Upsert documents to the vector index with source IDs instead of full content"""
        try:
            vectors = []
            
            # Collect all content for batch embedding
            contents = []
            doc_infos = []
            
            for doc in documents:
                content = doc.get("content", "")
                if not content:
                    continue
                    
                contents.append(content.replace("\n", " "))
                doc_infos.append(doc)
            
            if not contents:
                return False
            
            # Generate embeddings in batch using LiteLLM
            response = embedding(model=self.embedding_model, input=contents)
            embeddings = [data['embedding'] for data in response.data]
            
            # Create vectors with minimal metadata (just source ID and title)
            for doc, emb in zip(doc_infos, embeddings):
                vectors.append({
                    "id": doc.get("id", f"doc_{len(vectors)}"),
                    "values": emb,
                    "metadata": {
                        "source_id": doc.get("id"),  # Store source ID for database lookup
                        "title": doc.get("title", ""),
                        "source_type": doc.get("source_type", ""),
                        # Don't store full content here anymore
                        **doc.get("metadata", {})
                    }
                })
            
            if vectors:
                self.index.upsert(vectors=vectors)
                logger.info(f"Upserted {len(vectors)} vectors")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error upserting documents: {e}")
            return False
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the index"""
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {} 