# Library

## Overview

The Library subsystem generates a browsable, static HTML website from all knowledge base content. It features AI-powered theme generation, book cover resolution, and client-side fuzzy search. The generated site can be served locally or deployed statically.

## Key Components

| File | Purpose |
|------|---------|
| `src/library/models.py` | Data models: ContentType, LibraryItem, Theme, ThemeAssignment |
| `src/library/builder.py` | Orchestrates the 5-step build pipeline |
| `src/library/content_exporter.py` | Exports ChromaDB documents to LibraryItems |
| `src/library/theme_generator.py` | AI-powered theme generation and assignment |
| `src/library/cover_resolver.py` | Book cover lookup via Open Library and Google Books APIs |
| `src/library/search_indexer.py` | Generates Fuse.js search index |
| `src/library/static_generator.py` | Renders Jinja2 templates to HTML |
| `src/library/server.py` | Local HTTP server for previewing the site |
| `src/library/templates/` | Jinja2 HTML templates |
| `src/library/assets/` | CSS, JavaScript, and placeholder images |

## Data Models

### ContentType

Enumeration of content categories:

| Type | Description |
|------|-------------|
| `BOOK` | Books and long-form reading |
| `ARTICLE` | Articles and shorter written content |
| `FRAMEWORK` | Problem-solving frameworks and methodologies |
| `VALUE` | Personal values and principles |
| `PREFERENCE` | Communication and work style preferences |
| `MEMORY` | Stored memories from conversations |
| `INSIGHT` | Extracted insights and learnings |
| `GOAL` | Personal or project goals |
| `SKILL` | Skills and capabilities |
| `WEB_CONTENT` | Generic web content (URLs, pages) |

### LibraryItem

Represents a single piece of content:

```python
@dataclass
class LibraryItem:
    id: str                      # Unique identifier
    content_type: ContentType    # Category
    title: str                   # Display title
    summary: str                 # 200-char truncated summary
    full_content: str            # Complete content
    source_url: Optional[str]    # Original URL if applicable
    cover_image_url: Optional[str]  # Resolved cover image
    metadata: Dict[str, Any]     # Additional metadata
    themes: List[str]            # Assigned theme IDs
    created_at: datetime         # Creation timestamp
    highlights: List[str]        # Key points, steps, etc.
```

### Theme

AI-generated content category:

```python
@dataclass
class Theme:
    id: str                 # Generated slug
    name: str               # Display name
    description: str        # What this theme covers
    keywords: List[str]     # Search keywords
    item_count: int         # Number of assigned items
    created_at: datetime
    updated_at: datetime
```

### ThemeAssignment

Links items to themes with confidence:

```python
@dataclass
class ThemeAssignment:
    item_id: str
    theme_id: str
    confidence: float       # 0.0-1.0
    assigned_at: datetime
```

## Build Pipeline

The `LibraryBuilder.build()` method orchestrates a 5-step pipeline:

### Step 1: Content Export

`ContentExporter` transforms ChromaDB documents into LibraryItems:

- **Type Inference**: Maps source metadata to ContentType
  - `value` → VALUE, `framework` → FRAMEWORK, `memory` → MEMORY
  - URL sources → WEB_CONTENT or ARTICLE based on content
  - Fallback: WEB_CONTENT
- **Title Generation**: First 50 characters of content, cleaned
- **Summary Truncation**: 200 characters with ellipsis
- **Highlight Extraction**: Type-specific key points
  - VALUES: `key_points` field
  - FRAMEWORK: `steps` field
  - INSIGHT: `takeaways` field

### Step 2: Cover Resolution

`CoverResolver` finds book cover images:

- **ISBN Lookup**: Open Library API (`/isbn/{isbn}.json`)
- **Title Search**: Open Library search API, then Google Books API
- **Placeholders**: SVG icons for non-book content types
- **Caching**: Resolved URLs cached to `cover_cache.json`
- **Retry Logic**: Exponential backoff (3 attempts, max 30s delay)

### Step 3: Theme Generation

`ThemeGenerator` uses the LLM to categorize content:

- **Theme Discovery**: Samples content, prompts LLM to identify themes
- **Item Assignment**: Assigns each item to 1-3 themes with confidence scores
- **Miscellaneous Theme**: Catches items with confidence < 0.3
- **Persistence**: Saves to `themes.json` and `assignments.json`
- **Incremental Updates**: Only processes new/changed items

### Step 4: Search Index

`SearchIndexer` builds a Fuse.js-compatible index:

- **Field Weights**:
  - `title`: 1.0 (highest)
  - `keywords`: 0.8
  - `summary`: 0.6
  - `themes`: 0.4
- **Output**: `search-index.json` for client-side loading

### Step 5: Static Generation

`StaticGenerator` renders HTML:

- **Templates**: Jinja2 with autoescape enabled
- **Pages Generated**:
  - `index.html` - Home with theme cards and statistics
  - `all/index.html` - All content with sorting
  - `search/index.html` - Search interface
  - `theme/{id}/index.html` - Per-theme pages
  - `item/{id}/index.html` - Per-item detail pages
- **Assets**: Copies CSS, JS, and placeholder images

## Templates

| Template | Purpose |
|----------|---------|
| `base.html` | Layout with header, navigation, search form |
| `index.html` | Home page with theme cards and item counts |
| `theme.html` | Theme detail with breadcrumbs and item grid |
| `item.html` | Item detail with cover, metadata, highlights, source link |
| `all.html` | All content with sort dropdown |
| `search.html` | Search results page |
| `macros.html` | Reusable card rendering components |

## Frontend Assets

### CSS (`assets/css/styles.css`)

Design system with CSS custom properties:

| Property | Value |
|----------|-------|
| Background | `#FAFAFA` |
| Card | `#FFFFFF` |
| Text | `#1A1A1A` |
| Accent | `#2563EB` |

Responsive grid breakpoints:
- `>1024px`: 4 columns
- `>768px`: 3 columns
- `<768px`: 1-2 columns

Badge colors defined per ContentType.

### JavaScript (`assets/js/search.js`)

- Fuse.js library loaded from CDN
- Lazy-loads search index on first search
- 300ms input debounce
- Sort options: Recent, A-Z, By Type
- Load-more pagination (50 items per page)

### Placeholder Images (`assets/images/placeholders/`)

10 SVG icons, one per ContentType:
`book.svg`, `article.svg`, `framework.svg`, `value.svg`, `preference.svg`, `memory.svg`, `insight.svg`, `goal.svg`, `skill.svg`, `web_content.svg`

## HTTP Server

`LibraryServer` provides local preview:

- Uses Python's `http.server` module
- Auto-port fallback: tries 10 consecutive ports if primary is in use
- Thread-based non-blocking mode
- Default port: 8080 (configurable via `settings.library_server_port`)

## Build State Management

Incremental builds track state in `_build_state.json`:

```json
{
  "item_hashes": {
    "item_id_1": "sha256_of_content_fields",
    "item_id_2": "sha256_of_content_fields"
  },
  "last_build": "2024-01-15T10:30:00Z",
  "theme_version": "1.0"
}
```

- Hash computed from: id, content_type, title, summary, full_content, source_url, metadata, themes, highlights (SHA-256)
- Detects changed items by content hash comparison
- Skips regeneration for unchanged items
- `--force` flag rebuilds everything (also clears cover cache)

## Data Storage

| File | Contents |
|------|----------|
| `data/library/themes.json` | Generated themes |
| `data/library/assignments.json` | Item-to-theme mappings |
| `data/library/cover_cache.json` | Cached cover URLs |
| `data/library-site/` | Generated HTML site |
| `data/library-site/_build_state.json` | Build state tracking |

## Edge Cases

- **Empty knowledge base**: Generates site with "No content yet" message
- **API failures**: Falls back to placeholder images, logs warnings
- **Large datasets**: Pagination in search results (50 items per page)
- **Missing themes**: Items assigned to "Miscellaneous" catch-all theme
- **Duplicate content**: Deduplication by content hash
- **Unicode content**: Proper encoding in templates and JSON
- **Port conflicts**: Server tries 10 alternative ports before failing

## Dependencies

| Dependency | Purpose |
|------------|---------|
| Jinja2 | Template rendering |
| ChromaDB | Source data (via KnowledgeBase) |
| Ollama | Theme generation (llama3:8b) |
| requests | Cover API calls |
| Fuse.js (CDN) | Client-side search |
