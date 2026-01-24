# Document Ingestion

## Overview

Fetches web content, classifies it using LLM, and stores it in the knowledge base. Supports single URLs, batch concurrent processing, and manual text entry.

## Key Components

- `src/document_ingestion.py`: DocumentIngestionTool, BatchProgress, URLProcessingResult

## Data Structures

### ProcessingStatus
```python
class ProcessingStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
```

### URLProcessingResult
```python
@dataclass
class URLProcessingResult:
    url: str
    status: ProcessingStatus
    classification: Optional[Dict[str, Any]]
    error: Optional[str]
```

### BatchProgress
```python
@dataclass
class BatchProgress:
    total: int
    completed: int
    failed: int
    current_url: Optional[str]
    results: List[URLProcessingResult]

    @property
    def pending(self) -> int
    @property
    def percentage(self) -> float
```

## Content Types

LLM classifies content into:

| Type | Description |
|------|-------------|
| `value` | Core personal/organizational values or principles |
| `framework` | Problem-solving methodology, process, or structured approach |
| `preference` | Personal preferences, recommendations, or settings |
| `reference` | General reference material, facts, or documentation |

## Behavior

### Single URL Ingestion

```python
classification = ingestion_tool.add_web_content(
    url="https://example.com/article",
    user_context="For my productivity system"
)
```

1. Fetch webpage content
2. Handle Google Docs URLs (convert to export format)
3. Parse HTML, remove scripts/styles
4. Convert to markdown
5. Classify with LLM (type, name, category, description, key_points)
6. Store in knowledge base with metadata
7. Save to appropriate file if structured type

### Batch URL Processing

```python
progress = ingestion_tool.batch_process_urls_sync(
    urls=["https://url1.com", "https://url2.com"],
    user_context="Research materials",
    max_concurrent=5,
    progress_callback=update_ui
)
```

1. Create semaphore for concurrency limiting
2. Process URLs in parallel (up to `max_concurrent`)
3. Call progress_callback after each URL
4. Return BatchProgress with all results

### Text Content

```python
classification = ingestion_tool.add_text_content(
    content="My productivity principles...",
    user_context="Personal notes",
    content_name="Productivity Framework"
)
```

Same classification and storage flow, but:
- Metadata uses `source: "manual"`
- The provided `content_name` overrides the LLM's `name`
- Stored content uses `Source: Manual Entry`
- When saving to `knowledge/values.json`, `knowledge/preferences.json`, or `knowledge/frameworks/*.json`, the `source` field is set to `manual_entry`

### Google Docs Handling

URLs like `docs.google.com/document/d/{id}/edit` are converted to:
`docs.google.com/document/d/{id}/export?format=txt`

Returns plain text instead of HTML. Non-`/edit` Google Docs URLs are fetched as-is.

## Classification Response Format

```json
{
  "type": "framework",
  "name": "Getting Things Done",
  "category": "productivity",
  "description": "David Allen's productivity methodology",
  "key_points": [
    "Capture everything",
    "Clarify next actions",
    "Organize by context"
  ]
}
```

## File Storage

Classified content is saved to files:

| Type | Location |
|------|----------|
| `framework` | `knowledge/frameworks/{slugified_name}.json` (lowercase, spaces to underscores) |
| `value` | Appended to `knowledge/values.json` |
| `preference` | Added to `knowledge/preferences.json` |
| `reference` | Knowledge base only (no file) |

### Stored File Shapes

`knowledge/values.json`:
```json
{
  "values": [
    {
      "name": "Integrity",
      "description": "...",
      "source": "https://...",
      "user_context": "Personal",
      "key_points": ["...", "..."]
    }
  ]
}
```

`knowledge/preferences.json`:
```json
{
  "communication": {
    "Tone": {
      "description": "...",
      "source": "https://...",
      "user_context": "Personal",
      "points": ["...", "..."]
    }
  }
}
```

## Stored Content Format

```
Source: {url}
Context: {user_context}

{description}

Key Points:
- {point1}
- {point2}

Content:
{content (truncated to 3000 chars)}
```

For manual text ingestion, content is stored without truncation.

## Edge Cases

- **Classification fails**: Falls back to `type: "reference"`, `name: "Web Content"`, `category: "general"`, `description: "Content from web source"`
- **Fetch timeout**: 30 second timeout, logged as error
- **Invalid JSON in LLM response**: Regex extraction with fallback
- **Batch URL failure**: Counted in `failed`, error stored in result
- **Empty URLs list**: Returns immediately with 100% progress

## Dependencies

- `KnowledgeBase`: For document storage
- `OllamaClient`: For content classification
- `requests`: Synchronous HTTP
- `aiohttp`: Asynchronous HTTP
- `BeautifulSoup`: HTML parsing
- `markdownify`: HTML to markdown conversion
