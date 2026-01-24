# Memory and Context System

## Overview

The memory system provides persistent storage for memories and operating contexts. It enables project-specific work, recalls relevant information based on queries, and automatically extracts memories from conversations.

## Key Components

- `src/memory_system.py`: MemorySystem class, Memory and Context dataclasses

## Memory Types

Enum values are stored as lowercase strings:

| Type | Purpose |
|------|---------|
| `personal` | Personal info, preferences, background |
| `project` | Project-specific context and history |
| `insight` | Key insights and learnings |
| `goal` | Goals and objectives |
| `decision` | Important decisions made |
| `relationship` | People and relationships |
| `skill` | Skills and competencies |
| `experience` | Past experiences and lessons |

## Data Structures

### Memory
```python
@dataclass
class Memory:
    id: str                    # Unique identifier
    memory_type: MemoryType    # Category (see table above)
    content: str               # The memory content
    context: str               # Context/project this belongs to
    importance: int            # 1-10 scale
    created_at: datetime
    last_accessed: datetime
    access_count: int          # Tracks how often recalled
    tags: Set[str]             # Keywords for categorization
    projects: Set[str]         # Associated project IDs
    metadata: Dict[str, Any]
```

### Context
```python
@dataclass
class Context:
    id: str                    # Unique identifier (slug format)
    name: str                  # Display name
    description: str           # What this context is about
    context_type: str          # Category: work, personal, creative, etc.
    goals: List[str]           # Objectives for this context
    key_people: List[str]      # Relevant people
    current_focus: str         # Current focus area
    status: str                # active, paused, completed, archived
    created_at: datetime
    last_used: datetime
    metadata: Dict[str, Any]
```

## Storage

| File/Directory | Contents |
|----------------|----------|
| `data/memory/memories.json` | All memories (JSON) |
| `data/memory/contexts/*.json` | One file per context |
| `data/memory/settings.json` | Auto-switch settings |

### File Formats
`data/memory/memories.json` stores a JSON object keyed by memory ID:
```json
{
  "mem_20240115_143022": {
    "id": "mem_20240115_143022",
    "memory_type": "goal",
    "content": "Finish the quarterly report",
    "context": "work",
    "importance": 7,
    "created_at": "2024-01-15T14:30:22",
    "last_accessed": "2024-01-15T14:30:22",
    "access_count": 1,
    "tags": ["report", "deadline"],
    "projects": ["work"],
    "metadata": {}
  }
}
```

## Default Contexts

Created on first run:
- `work` - Professional work, career, business tasks
- `personal` - Personal life, family, friends, health
- `creative` - Creative projects, art, writing, music
- `reflection` - Self-reflection, personal growth, philosophy
- `personal_brand` - Building personal brand, content creation
- `product_dev` - Product development and improvement (context_type: `product_development`)
- `self_improvement` - Personal development, learning, skills

## Behavior

### Adding Memories

1. Generate unique ID with timestamp
2. Associate with current context (or "general")
3. Save to `memories.json`
4. Add to knowledge base for semantic search

### Recalling Memories

1. Build filter for context and memory types
2. Search knowledge base semantically
3. Match results to Memory objects
4. Update access count and timestamp
5. Return sorted by relevance

### Context Switching

1. Validate context ID exists
2. Set as `current_context`
3. Update `last_used` timestamp
4. Save context to disk

### Context Retrieval for Queries

```python
context_info = memory_system.get_context_for_query("How should I prioritize this?")
# Returns:
# {
#   "current_context": "<markdown summary>" | "No context selected",
#   "relevant_memories": [
#     {"type": "goal", "content": "...", "importance": 7, "context": "work"},
#     ...
#   ],
#   "context_history": ["2024-01-15: ...", ...]
# }
```

- `current_context` is a formatted summary string (or "No context selected")
- `relevant_memories` is derived from semantic search and includes type/content/importance/context
- `context_history` is always present; it is empty when no current context exists and otherwise populated from recent memories

### Memory Extraction from Conversations

Uses LLM to analyze conversation text and extract:
- Type, content, importance (1-10), and tags
- Returns JSON array of extracted memories
- Each is added via `add_memory()`
- The extractor prompt uses uppercase labels (e.g., `PERSONAL`) but they are normalized to lowercase enum values on ingest

## Auto-Switch Settings

Stored in `settings.json`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `enabled` | `true` | Whether auto-switching is active |
| `auto_switch_threshold` | `0.8` | Confidence for silent switch |
| `prompt_threshold` | `0.6` | Confidence for asking user |
| `switch_history` | `[]` | Last 100 context switches |

## Context Summary Format

```markdown
**Context: [Name]**

**Description**: [description]

**Current Focus**: [current_focus]

**Goals**:
- [goal 1]
- [goal 2]

**Recent Context**:
- [memory_type]: [content]...

**Status**: [status]
**Last Used**: [timestamp]
```

## Edge Cases

- **No current context**: Falls back to "general" for memories
- **Duplicate context ID**: Appends counter suffix (e.g., `my_project_1`)
- **Missing memory file**: Returns empty dict, creates on first save
- **Invalid memory type in extraction**: Logged and skipped
- **Empty conversation text**: Returns empty list of memories

## Dependencies

- `KnowledgeBase`: For semantic search of memories
- `OllamaClient`: For memory extraction from conversations
- `settings.conversation_history_dir.parent`: Base path for storage
