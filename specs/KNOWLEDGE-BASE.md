# Knowledge Base

## Overview

The knowledge base provides persistent semantic search using ChromaDB and Ollama embeddings. It stores values, frameworks, preferences, and memories, enabling context-aware retrieval for agent responses.

## Key Components

- `src/knowledge_base.py`: KnowledgeBase class
- ChromaDB: Vector database for embeddings
- Ollama: Embedding generation via `nomic-embed-text` model

## Storage

| Location | Contents |
|----------|----------|
| `data/chroma/` | ChromaDB persistent storage |
| `knowledge/values.json` | Personal values and principles |
| `knowledge/preferences.json` | Work and communication preferences |
| `knowledge/frameworks/*.json` | Problem-solving frameworks |

## Document Types

| Type | Source | Purpose |
|------|--------|---------|
| `value` | `knowledge/values.json` | Personal values and principles |
| `framework` | `knowledge/frameworks/*.json` | Problem-solving methodologies |
| `preference` | `knowledge/preferences.json` | Work style and preferences |
| `memory` | Runtime | Extracted from conversations |
| `quote` | Quotes system | Inspirational quotes for semantic search |
| `reference` | Document ingestion | General reference material |

## Behavior

### Adding Documents

```python
knowledge_base.add_document(
    content="Document text",
    metadata={"type": "value", "name": "Integrity"},
    document_id="optional_custom_id"
)
```

1. Generate embedding via Ollama
2. Store in ChromaDB with metadata
3. Auto-generate ID if not provided: `doc_{count}`

### Searching

```python
results = knowledge_base.search(
    query="How should I approach this decision?",
    n_results=5,
    filter_metadata={"type": "value"}
)
```

1. Generate query embedding
2. Query ChromaDB with cosine similarity
3. Apply metadata filters if provided (passed directly to ChromaDB `where`, so either simple equality or `$eq` works)
4. Return list of matches with content, metadata, distance, and ID

### Search Result Format

```python
{
    "content": str,      # Document text
    "metadata": dict,    # Stored metadata
    "distance": float,   # Cosine distance (lower = more similar)
    "id": str           # Document ID
}
```

## Loading Knowledge Files

On initialization, loads from `knowledge/` directory:

1. **Values** (`values.json`):
   - Parses `values` array
   - Builds content from: name, description, key_points, source, user_context
   - Tags with `type: "value"`, `source: "knowledge_files"`

2. **Frameworks** (`frameworks/*.json`):
   - Parses each JSON file
   - Content includes: name, description, steps
   - Tags with `type: "framework"`, `category`, `source: "knowledge_files"`

3. **Preferences** (`preferences.json`):
   - Iterates category/preference pairs
   - Preference values may be simple strings or nested objects; nested objects are stored as their string representation
   - Tags with `type: "preference"`, `category`, `source: "knowledge_files"`

Skips loading if `source: "knowledge_files"` documents already exist.

## Export/Import

### Export

```python
export_data = knowledge_base.export_knowledge(
    filter_type="value",       # Optional
    filter_source="web"        # Optional
)
```

Returns:
```python
{
    "export_version": "1.0",
    "export_timestamp": "ISO timestamp",
    "collection_name": str,
    "total_documents": int,
    "filters": {"type": str, "source": str},
    "documents": [
        {
            "id": str,
            "content": str,
            "metadata": dict,
            "embedding": list  # Optional
        }
    ]
}
```

### Import

```python
stats = knowledge_base.import_knowledge(
    import_data=export_dict,
    duplicate_handling="skip"  # or "update" or "fail"
)
```

Duplicate handling:
- `skip`: Ignore documents with existing IDs
- `update`: Delete and re-add
- `fail`: Raise ValueError on duplicate

Note: With `duplicate_handling="update"`, the current implementation increments the `updated` counter for every imported document (even when there was no duplicate), so treat `updated` as a rough "processed" count in that mode.

Returns:
```python
{
    "total_in_file": int,
    "added": int,
    "updated": int,
    "skipped": int,
    "errors": int,
    "error_messages": list
}
```

## Statistics

```python
stats = knowledge_base.get_stats()
# Returns:
# {
#     "total_documents": int,
#     "types": {"value": 5, "framework": 3, ...}
# }
```

## Edge Cases

- **Missing embedding in import**: Regenerated via Ollama
- **Missing ID or content in import**: Skipped, counted as error
- **Duplicate ID with skip**: Silently skipped
- **Empty knowledge directory**: No documents loaded, no error
- **ChromaDB connection failure**: Exception propagated

## Dependencies

- `chromadb`: Vector database
- `OllamaClient`: For embedding generation
- `settings.chroma_persist_directory`: Storage path
- `settings.chroma_collection_name`: Collection name
