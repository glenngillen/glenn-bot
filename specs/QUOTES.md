# Quotes System

## Overview

Manages inspirational quotes with AI-powered categorization, semantic search, and integration with the memory system for reflection and inspiration.

## Key Components

- `src/quotes_system.py`: QuotesSystem class and Quote dataclass

## Data Structures

### Quote
```python
@dataclass
class Quote:
    id: str                         # Unique identifier
    text: str                       # The quote text
    author: str                     # Who said it
    source: Optional[str]           # Where it came from
    context: str                    # Why this quote resonates
    tags: Set[str]                  # Keywords for categorization
    category: str                   # inspiration, wisdom, leadership, etc.
    created_at: datetime
    last_reflected: Optional[datetime]
    reflection_count: int           # Times reflected on
    importance: int                 # 1-10 scale
    projects: Set[str]              # Associated context IDs
```

## Categories

- `inspiration`
- `wisdom`
- `leadership`
- `creativity`
- `productivity`
- `success`
- `relationships`
- `growth`
- `courage`
- `innovation`
- `life`
- `work`
- `entrepreneurship`

## Storage

| File | Contents |
|------|----------|
| `data/quotes/quotes.json` | All quotes |

### File Format
```json
{
  "quote_0_20240201_103022": {
    "id": "quote_0_20240201_103022",
    "text": "The only way to do great work is to love what you do.",
    "author": "Steve Jobs",
    "source": "Stanford Commencement",
    "context": "Reminder about passion in work",
    "tags": ["work", "passion", "success"],
    "category": "inspiration",
    "created_at": "2024-02-01T10:30:22",
    "last_reflected": null,
    "reflection_count": 0,
    "importance": 8,
    "projects": ["work"]
  }
}
```

## Behavior

### Adding Quotes

```python
quote = quotes_system.add_quote(
    text="The only way to do great work is to love what you do.",
    author="Steve Jobs",
    context="Reminder about passion in work",
    source="Stanford Commencement",
    category="inspiration",
    importance=8,
    tags={"work", "passion", "success"}
)
```

1. Generate unique ID with timestamp
2. Associate with current context (or `general` if none)
3. Save to quotes.json
4. Add to knowledge base for semantic search (`type: "quote"`, `source: "quotes_system"`)
5. Create memory entry

### AI Categorization

```python
suggestion = quotes_system.categorize_quote(
    text="Quote text...",
    author="Author name",
    context="Why I saved this"
)
# Returns:
# {
#     "category": "wisdom",
#     "tags": ["growth", "mindset", "learning"],
#     "importance": 7,
#     "explanation": "Why this quote is valuable..."
# }
```

### Getting Quotes for Reflection

```python
quote = quotes_system.get_random_quote(
    category="wisdom",         # Optional filter
    context_filter="work"      # Optional project filter
)
```

Selection logic:
1. Filter by category/context if specified
2. Prefer quotes never reflected on
3. Weight remaining by days since last reflection
4. Update `last_reflected` and `reflection_count`

### Searching Quotes

```python
# Semantic search
quotes = quotes_system.search_quotes("perseverance challenges", limit=5)

# By category
quotes = quotes_system.get_quotes_by_category("leadership")

# By author
quotes = quotes_system.get_quotes_by_author("Einstein")
```

### Reflection Prompt

```python
prompt = quotes_system.get_reflection_prompt(quote)
```

Generates structured reflection including:
- The quote and author
- Why you saved it
- Current context
- Reflection questions

## Statistics

```python
stats = quotes_system.get_stats()
# Returns:
# {
#     "total_quotes": 25,
#     "categories": {"wisdom": 10, "inspiration": 8, ...},
#     "top_authors": {"Marcus Aurelius": 5, ...},
#     "average_importance": 6.5,
#     "recent_quotes": 3,  # Last 7 days
#     "total_reflections": 42
# }
```

When there are no quotes, `get_stats()` returns `{ "total_quotes": 0 }`.

## Edge Cases

- **AI categorization fails**: Falls back to `category: "inspiration"`, `importance: 5`
- **No quotes match filters**: `get_random_quote()` returns None
- **Empty quotes file**: `get_stats()` returns `{ "total_quotes": 0 }`
- **Missing quote_id in search**: Skipped from results

## Dependencies

- `MemorySystem`: For creating memory entries
- `KnowledgeBase`: For semantic search storage
- `OllamaClient`: For AI categorization
