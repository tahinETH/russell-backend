#!/usr/bin/env python3
"""
Document loading script for Pinecone
"""
import json
import sys
import os
import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
import tiktoken
from litellm import embedding
from pinecone import Pinecone, ServerlessSpec

class DocumentLoader:
    def __init__(self):
        """Initialize the DocumentLoader with Pinecone connection and necessary components."""
        # Initialize Pinecone
        self.pc = Pinecone(
            api_key=settings.pinecone_api_key, 
            environment=settings.pinecone_environment,
            name=settings.pinecone_index_name
        )
        
        # Initialize tokenizer for text chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Configuration from settings
        self.embedding_model = settings.embedding_model
        self.chunk_size = settings.chunk_size  # Use chunk_size from config
        self.embedding_ctx_length = settings.embedding_ctx_length
        
        # Batch sizes for processing
        self.EMBEDDING_BATCH_SIZE = 100
        self.UPSERT_BATCH_SIZE = 100
        
        print(f"Initialized DocumentLoader with:")
        print(f"  - Embedding model: {self.embedding_model}")
        print(f"  - Chunk size: {self.chunk_size} tokens")
        print(f"  - Context length: {self.embedding_ctx_length}")

    async def initialize(self):
        """Async initialization method to set up the index."""
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
                # Get embedding dimension using the configured model
                response = embedding(model=self.embedding_model, input=["test"])
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

    def chunk_text(self, text: str, overlap: int = 50) -> List[str]:
        """Split text into chunks based on token count with overlap."""
        tokens = self.tokenizer.encode(text)
        chunks = []

        while len(tokens) > 0:
            # Take up to chunk_size tokens
            chunk_tokens = tokens[:self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move forward, considering overlap
            if len(tokens) <= self.chunk_size:
                break
            tokens = tokens[self.chunk_size - overlap:]
            
        return chunks

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts.""" 
        try:
            # Clean texts
            texts = [text.replace("\n", " ") for text in texts]
            
            # Process in batches
            all_embeddings = []
            for i in range(0, len(texts), self.EMBEDDING_BATCH_SIZE):
                batch = texts[i:i + self.EMBEDDING_BATCH_SIZE]
                embedding_start = time.time()
                
                response = embedding(model=self.embedding_model, input=batch)
                batch_embeddings = [data['embedding'] for data in response.data]
                
                embedding_time = time.time() - embedding_start
                print(f"Embedding generation for {len(batch)} texts took {embedding_time:.2f}s ({len(batch)/embedding_time:.1f} texts/s)")
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting
                await asyncio.sleep(0.1)
            
            return all_embeddings
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)

    async def _process_document_batch(self, documents: List[Dict]) -> List[tuple]:
        """Process a batch of documents and return vectors to upsert."""
        chunk_start = time.time()
        all_chunks = []
        chunk_metadata = []
        
        # Process each document into chunks
        for doc in documents:
            doc_id = doc.get('id', str(uuid.uuid4()))
            title = doc.get('title', '')
            content = doc.get('content', '')
            link = doc.get('link', '')
            
            if not content.strip():
                continue

            # Get chunks of the document content
            chunks = self.chunk_text(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "doc_id": doc_id,
                    "chunk_index": chunk_idx,
                    "title": title,
                    "link": link,
                    "content": chunk,
                    "source": "faq" if doc_id.startswith("faq_") else "scientific_papers"
                })

        chunk_time = time.time() - chunk_start
        print(f"Chunking {len(documents)} documents took {chunk_time:.2f}s")

        if not all_chunks:
            return []

        # Get embeddings for all chunks in batches
        embeddings = await self.get_embeddings_batch(all_chunks)
        
        # Create vectors for valid embeddings
        vector_start = time.time()
        vectors = []
        for chunk, embedding_vec, metadata in zip(all_chunks, embeddings, chunk_metadata):
            if embedding_vec is not None:
                vector_id = f"{metadata['doc_id']}_chunk_{metadata['chunk_index']}"
                vectors.append((vector_id, embedding_vec, metadata))
        
        vector_time = time.time() - vector_start
        print(f"Vector creation took {vector_time:.2f}s")

        return vectors

    async def _upsert_batch(self, vectors: List[tuple]):
        """Upsert vectors to Pinecone in batches."""
        try:
            for i in range(0, len(vectors), self.UPSERT_BATCH_SIZE):
                batch = vectors[i:i + self.UPSERT_BATCH_SIZE]
                upsert_start = time.time()
                self.index.upsert(vectors=batch)
                upsert_time = time.time() - upsert_start
                print(f"Pinecone upsert of {len(batch)} vectors took {upsert_time:.2f}s ({len(batch)/upsert_time:.1f} vectors/s)")
                await asyncio.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"Error upserting batch to Pinecone: {e}")

    async def load_documents_from_json(self, json_path: str) -> List[Dict]:
        """Load documents from JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            print(f"Loaded {len(documents)} documents from {json_path}")
            return documents
        except Exception as e:
            print(f"Error loading documents from {json_path}: {e}")
            return []

    async def load_and_process_documents(self, json_files: List[str] = None):
        """Load and process documents from JSON files into Pinecone."""
        if json_files is None:
            # Default to faq.json and scientific_papers.json
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_files = [
                os.path.join(script_dir, "faq.json"),
                os.path.join(script_dir, "scientific_papers.json")
            ]
        
        all_documents = []
        
        # Load documents from all JSON files
        for json_file in json_files:
            if os.path.exists(json_file):
                docs = await self.load_documents_from_json(json_file)
                all_documents.extend(docs)
            else:
                print(f"Warning: File {json_file} not found")
        
        if not all_documents:
            print("No documents to process")
            return False
        
        print(f"\nProcessing {len(all_documents)} total documents...")
        start_time = time.time()
        
        vectors_to_upsert = []
        total_processed = 0
        total_chunks = 0
        
        # Process documents in batches
        batch_size = 50  # Process 50 documents at a time
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i + batch_size]
            batch_start = time.time()
            
            # Process entire batch at once
            vectors = await self._process_document_batch(batch)
            
            # Update statistics
            vectors_to_upsert.extend(vectors)
            total_chunks += len(vectors)
            
            # Upsert vectors when we have enough or on last batch
            if len(vectors_to_upsert) >= self.UPSERT_BATCH_SIZE or i + batch_size >= len(all_documents):
                await self._upsert_batch(vectors_to_upsert)
                vectors_to_upsert = []
            
            total_processed += len(batch)
            batch_time = time.time() - batch_start
            
            # Progress logging
            print(f"\nBatch {i//batch_size + 1} stats:")
            print(f"- Processed {len(batch)} documents in {batch_time:.2f}s")
            print(f"- Generated {len(vectors)} chunks")
            print(f"- Progress: {total_processed}/{len(all_documents)} documents ({(total_processed/len(all_documents)*100):.1f}%)")
            print("-" * 50)

        total_time = time.time() - start_time
        
        print(f"\nProcessing complete:")
        print(f"Time taken: {total_time:.2f}s")
        print(f"Average speed: {total_processed/total_time:.1f} documents/s")
        print(f"Total documents processed: {total_processed}")
        print(f"Total chunks created: {total_chunks}")
        print(f"Final index stats: {self.index.describe_index_stats()}")
        
        return True

async def main():
    """Main function to load documents into Pinecone"""
    print("=== Enhanced Document Loading Script ===")
    
    # Initialize document loader
    try:
        loader = DocumentLoader()
        await loader.initialize()
        print("Document loader initialized successfully")
    except Exception as e:
        print(f"Failed to initialize document loader: {e}")
        return
    
    # Load and process documents
    if len(sys.argv) > 1:
        # Load from specified JSON files
        json_files = sys.argv[1:]
        success = await loader.load_and_process_documents(json_files)
    else:
        # Use default files (faq.json and scientific_papers.json)
        print("Loading default files: faq.json and scientific_papers.json")
        success = await loader.load_and_process_documents()
    
    if success:
        print("Documents uploaded successfully!")
    else:
        print("Failed to upload documents")
    
    print("=== Document Loading Complete ===")

def print_usage():
    """Print usage instructions"""
    print("Usage:")
    print("  python scripts/load_documents.py [json_file1] [json_file2] ...")
    print("")
    print("Examples:")
    print("  python scripts/load_documents.py                           # Load faq.json and scientific_papers.json")
    print("  python scripts/load_documents.py faq.json                  # Load only faq.json")
    print("  python scripts/load_documents.py faq.json scientific_papers.json  # Load both files")
    print("")
    print("JSON format:")
    print("""[
  {
    "id": "doc_001",
    "title": "Document Title",
    "content": "Document content here...",
    "link": "https://example.com/link"
  }
]""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print_usage()
    else:
        asyncio.run(main()) 