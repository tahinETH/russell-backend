from pinecone import Pinecone
from litellm import embedding
from typing import List, Dict
import asyncio
import logging

logger = logging.getLogger(__name__)

class VectorService:
    def __init__(self, api_key: str, environment: str, index_name: str, embedding_model: str = "text-embedding-3-large"):
        try:
            # Use new Pinecone class instead of pinecone.init()
            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index(index_name)
            self.embedding_model = embedding_model
            logger.info(f"Initialized VectorService with index: {index_name} and model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize VectorService: {e}")
            raise
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents using vector similarity"""
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
            
            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "content": match.metadata.get("content", ""),
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    async def upsert_documents(self, documents: List[Dict]) -> bool:
        """Upsert documents to the vector index"""
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
            
            # Create vectors
            for doc, emb in zip(doc_infos, embeddings):
                vectors.append({
                    "id": doc.get("id", f"doc_{len(vectors)}"),
                    "values": emb,
                    "metadata": {
                        "content": doc.get("content", ""),
                        "title": doc.get("title", ""),
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