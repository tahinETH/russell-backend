import json
import time
import uuid
import asyncio
import re
from typing import List, Dict, Any, Optional

from litellm import embedding
from pinecone import Pinecone, ServerlessSpec
import tiktoken

from config import settings


class EmbeddingService:
    def __init__(self, model_name: str = None, use_pinecone: bool = True):
       
        self.use_pinecone = use_pinecone
        self.model_name = model_name or settings.embedding_model
        
        if self.use_pinecone:
            # Initialize Pinecone
            self.pc = Pinecone(
                api_key=settings.pinecone_api_key, 
                environment=settings.pinecone_environment
            )
            # Initialize tokenizer for text chunking
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            raise ValueError("Local embedding models are no longer supported. Please use use_pinecone=True with LiteLLM models.")
        
        # Batch size for embeddings and upserts
        self.BATCH_SIZE = 100
        self.CHUNK_SIZE = 500  # tokens
        self.UPSERT_BATCH_SIZE = 100

    async def initialize(self):
        """Async initialization method to set up the Pinecone index."""
        if not self.use_pinecone:
            return
            
        # Ensure index exists
        await self._setup_index()
        
        # Connect to the index
        self.index = self.pc.Index(settings.pinecone_index_name)
        print(f"Connected to Pinecone index: {settings.pinecone_index_name}")
        print(f"Index stats: {self.index.describe_index_stats()}")

    async def _setup_index(self):
        """Set up Pinecone index if it doesn't exist."""
        index_exists = any(index.name == settings.pinecone_index_name for index in self.pc.list_indexes())
        
        if not index_exists:
            print(f"Creating Pinecone index: {settings.pinecone_index_name}")
            try:
                # Get embedding dimension using LiteLLM
                response = embedding(model=settings.embedding_model, input=["test"])
                embedding_dim = len(response.data[0]['embedding'])
                print(f"Detected embedding dimension: {embedding_dim}")
            except Exception as e:
                print(f"Error getting embedding dimension: {e}. Defaulting to 1536.")
                embedding_dim = 1536

            # Create the index
            self.pc.create_index(
                settings.pinecone_index_name,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            # Wait for index to be ready
            while not self.pc.describe_index(settings.pinecone_index_name).status['ready']:
                print("Waiting for index to be ready...")
                await asyncio.sleep(1)
            print(f"Index {settings.pinecone_index_name} created and ready.")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using LiteLLM"""
        text = text.replace("\n", " ")
        try:
            response = embedding(model=settings.embedding_model, input=[text])
            return response.data[0]['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return asyncio.run(self.get_embeddings_batch(texts))

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts using LiteLLM.""" 
        try:
            # Clean texts
            texts = [text.replace("\n", " ") for text in texts]
            
            # Process in batches
            all_embeddings = []
            for i in range(0, len(texts), self.BATCH_SIZE):
                batch = texts[i:i + self.BATCH_SIZE]
                embedding_start = time.time()
                
                # Use LiteLLM for embeddings
                response = embedding(model=settings.embedding_model, input=batch)
                batch_embeddings = [data['embedding'] for data in response.data]
                
                embedding_time = time.time() - embedding_start
                print(f"Embedding generation for {len(batch)} texts took {embedding_time:.2f}s ({len(batch)/embedding_time:.1f} texts/s)")
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)

    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = 50) -> List[str]:
        """
        Split text into chunks with optional overlap using token-based chunking.
        
        Args:
            text: Input text to chunk
            chunk_size: Target size of each chunk in tokens
            overlap: Number of tokens to overlap between chunks
        
        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.CHUNK_SIZE
            
        # Token-based chunking
        tokens = self.tokenizer.encode(text)
        chunks = []

        while len(tokens) > 0:
            chunk_tokens = tokens[:chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            tokens = tokens[chunk_size - overlap:]  # Apply overlap
            
        return chunks

    async def prepare_documents_for_indexing(
        self, 
        documents: List[Dict], 
        chunk_size: int = None,
        overlap: int = 50
    ) -> List[Dict]:
        """
        Prepare documents for vector indexing by chunking and adding metadata.
        
        Args:
            documents: List of documents with 'content', 'id', and optional 'title'
            chunk_size: Target chunk size 
            overlap: Overlap between chunks
        
        Returns:
            List of prepared document chunks with metadata
        """
        prepared_docs = []
        
        for doc in documents:
            content = doc.get('content', '')
            doc_id = doc.get('id', f'doc_{len(prepared_docs)}')
            title = doc.get('title', '')
            
            if not content:
                continue
            
            chunks = self.chunk_text(content, chunk_size, overlap)
            
            for i, chunk in enumerate(chunks):
                prepared_docs.append({
                    'id': f"{doc_id}_chunk_{i}",
                    'content': chunk,
                    'metadata': {
                        'document_id': doc_id,
                        'title': title,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        **doc.get('metadata', {})
                    }
                })
        
        return prepared_docs

    async def store_documents(self, documents: List[Dict]):
        """
        Process documents and store embeddings in Pinecone.
        
        Args:
            documents: List of document dictionaries with 'content', 'id', and optional metadata
        """
        if not self.use_pinecone:
            raise ValueError("Pinecone storage requires use_pinecone=True")
            
        print(f"\nPreparing {len(documents)} documents for indexing...")
        prepared_docs = await self.prepare_documents_for_indexing(documents)
        print(f"Prepared documents: {len(prepared_docs)} chunks")

        print(f"\nProcessing {len(prepared_docs)} document chunks...")
        start_time = time.time()
        vectors_to_upsert = []
        total_processed = 0
        
        # Process documents in batches
        for i in range(0, len(prepared_docs), self.UPSERT_BATCH_SIZE):
            batch = prepared_docs[i:i + self.UPSERT_BATCH_SIZE]
            batch_start = time.time()
            
            # Get content and create embeddings
            texts = [doc['content'] for doc in batch]
            embeddings = await self.get_embeddings_batch(texts)
            
            # Create vectors for valid embeddings
            for doc, embedding in zip(batch, embeddings):
                if embedding is not None:
                    vector_id = doc['id']
                    metadata = doc['metadata'].copy()
                    metadata['content'] = doc['content']
                    vectors_to_upsert.append((vector_id, embedding, metadata))
            
            # Upsert when we have enough vectors or on last batch
            if len(vectors_to_upsert) >= self.UPSERT_BATCH_SIZE or i + self.UPSERT_BATCH_SIZE >= len(prepared_docs):
                await self._upsert_batch(vectors_to_upsert)
                vectors_to_upsert = []
            
            total_processed += len(batch)
            batch_time = time.time() - batch_start
            
            print(f"Batch {i//self.UPSERT_BATCH_SIZE + 1}: Processed {len(batch)} chunks in {batch_time:.2f}s")
            print(f"Progress: {total_processed}/{len(prepared_docs)} chunks ({(total_processed/len(prepared_docs)*100):.1f}%)")

        total_time = time.time() - start_time
        print(f"\nDocument storage complete:")
        print(f"Time taken: {total_time:.2f}s")
        print(f"Total chunks processed: {total_processed}")
        print(f"Final index stats: {self.index.describe_index_stats()}")

    async def _upsert_batch(self, vectors: List[tuple]):
        """Upsert vectors to Pinecone in batches."""
        try:
            for i in range(0, len(vectors), self.BATCH_SIZE):
                batch = vectors[i:i + self.BATCH_SIZE]
                upsert_start = time.time()
                self.index.upsert(vectors=batch)
                upsert_time = time.time() - upsert_start
                print(f"Pinecone upsert of {len(batch)} vectors took {upsert_time:.2f}s ({len(batch)/upsert_time:.1f} vectors/s)")
                await asyncio.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"Error upserting batch to Pinecone: {e}")

    async def search_similar_documents(
        self, 
        query: str, 
        top_k: int = 5, 
        filter_dict: Dict = None
    ) -> List[Dict]:
        """
        Search for similar documents in Pinecone.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of similar documents with scores
        """
        if not self.use_pinecone:
            raise ValueError("Document search requires use_pinecone=True")
            
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        if query_embedding is None:
            return []
        
        # Search in Pinecone
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
            
            # Format results
            documents = []
            for match in results['matches']:
                documents.append({
                    'id': match['id'],
                    'score': match['score'],
                    'content': match['metadata'].get('content', ''),
                    'metadata': {k: v for k, v in match['metadata'].items() if k != 'content'}
                })
            
            return documents
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []


# Backward compatibility functions - Updated to use LiteLLM
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Legacy function for chunking text using tiktoken"""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    tokens = tokenizer.encode(text)
    chunks = []

    while len(tokens) > 0:
        chunk_tokens = tokens[:chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        tokens = tokens[chunk_size - overlap:]
        
    return chunks

def prepare_documents_for_indexing(
    documents: List[Dict], 
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Dict]:
    """Legacy function for preparing documents"""
    service = EmbeddingService(use_pinecone=True)
    return asyncio.run(service.prepare_documents_for_indexing(documents, chunk_size, overlap)) 