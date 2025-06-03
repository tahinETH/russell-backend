# Vector Search Improvement: Database-Backed Content Retrieval

## Overview

The vector search system has been improved to address the issue of truncated content in search results. Previously, the system stored full content in Pinecone vector metadata, which led to truncated and incomplete information being passed to the LLM. The new approach stores only reference IDs in the vector index and retrieves full content from a dedicated database table.

## Architecture Changes

### Before
- **Vector Index**: Stored embeddings + full content in metadata
- **Issue**: Content was truncated, leading to incomplete context for LLM

### After
- **Vector Index**: Stores embeddings + source IDs in metadata
- **Sources Database Table**: Stores full content indexed by source ID
- **Search Process**: Vector search → Source IDs → Database lookup → Full content

## New Components

### 1. Sources Table
```sql
CREATE TABLE sources (
    id VARCHAR PRIMARY KEY,           -- e.g., "faq_001", "paper_001"
    title VARCHAR NOT NULL,
    content TEXT NOT NULL,            -- Full, untruncated content
    link VARCHAR,
    source_type VARCHAR NOT NULL,     -- "faq" or "scientific_papers"
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 2. SourceService
- `get_source_by_id()`: Retrieve single source
- `get_sources_by_ids()`: Retrieve multiple sources efficiently
- `bulk_create_sources()`: Populate sources from JSON files
- `populate_from_json_files()`: Load FAQ and scientific papers data

### 3. Enhanced VectorService
- Modified `search()`: Now fetches full content from database using source IDs
- Modified `upsert_documents()`: Stores only source IDs, not full content
- Added database integration via SourceService

## Setup Process

Run the comprehensive setup script:

```bash
python scripts/setup_sources_and_vectors.py
```

This script:
1. Creates the sources table
2. Populates it with data from `faq.json` and `scientific_papers.json`
3. Re-indexes vectors with new metadata structure (source IDs only)
4. Tests the new search functionality

## Benefits

1. **Complete Content**: LLM receives full, untruncated content
2. **Efficient Storage**: Vector index only stores necessary metadata
3. **Scalable**: Database can handle large content efficiently
4. **Maintainable**: Clear separation between search and content storage
5. **Flexible**: Easy to add new content types and sources

## Files Added/Modified

### New Files
- `app/models.py`: Added `Source` model and `SourceResponse` Pydantic model
- `app/services/source_service.py`: Source management service
- `scripts/populate_sources.py`: Populate sources table
- `scripts/reindex_vectors.py`: Re-index vectors with new structure
- `scripts/create_sources_table.py`: Create database table
- `scripts/setup_sources_and_vectors.py`: Comprehensive setup script

### Modified Files
- `app/services/vector.py`: Enhanced to use database for content retrieval

## Usage

The vector search now works transparently with the new architecture:

```python
# This now returns full content from database
results = await vector_service.search("What is precommitment?", top_k=5)

for result in results:
    print(f"Title: {result['metadata']['title']}")
    print(f"Full content: {result['content']}")  # Complete, untruncated
    print(f"Source: {result['metadata']['source_type']}")
```

## Data Sources

Currently populated with:
- **FAQ**: 38 entries from `scripts/faq.json`
- **Scientific Papers**: 33 entries from `scripts/scientific_papers.json`
- **Total**: 71 sources with complete content available

## Testing

The system includes comprehensive testing to verify:
- Database population success
- Vector indexing with correct metadata
- Search functionality returning full content
- End-to-end query processing 