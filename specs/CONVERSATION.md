# Conversation Management

## Overview

Manages conversation history with timestamped messages, persistence to disk, and retrieval of context for LLM interactions and feedback tracking.

## Key Components

- `src/conversation.py`: ConversationManager class and Message dataclass

## Data Structures

### Message
```python
@dataclass
class Message:
    role: str           # "user", "assistant", "system"
    content: str        # Message text
    timestamp: datetime
    metadata: Dict[str, Any]  # Optional additional data
```

## Storage

| Location | Format |
|----------|--------|
| `data/conversations/{id}.json` | One file per conversation |

### File Format
```json
{
  "id": "20240115_143022",
  "started_at": "2024-01-15T14:30:22",
  "messages": [
    {
      "role": "user",
      "content": "...",
      "timestamp": "2024-01-15T14:30:22",
      "metadata": {}
    }
  ]
}
```

## Behavior

### Adding Messages

```python
index = conversation_manager.add_message(
    role="user",
    content="Hello",
    metadata={"source": "terminal"}
)
```

1. Create Message with current timestamp
2. Append to `current_conversation`
3. Save to disk
4. Return message index (for feedback system)

### Getting Context

**For display** (excludes system messages):
```python
context = conversation_manager.get_conversation_context(max_messages=10)
# Returns:
# "User: Hello\n\nAssistant: Hi there!"
```

**For LLM**:
```python
messages = conversation_manager.get_messages_for_llm(max_messages=10)
# Returns:
# [{"role": "user", "content": "Hello"}, ...]
```

### Message Pair Retrieval (for Feedback)

```python
pair = conversation_manager.get_message_pair_for_feedback()
# Returns:
# {
#     "user_index": 0,
#     "user_content": "Hello",
#     "assistant_index": 1,
#     "assistant_content": "Hi there!"
# }
```

Finds the last assistant message and its preceding user message. Returns `None` if no assistant message exists.

### Conversation Lifecycle

**New conversation**:
```python
conversation_manager.start_new_conversation()
# - Clears current_conversation
# - Generates new ID from timestamp
```

**Load existing**:
```python
conversation_manager.load_conversation("20240115_143022")
# - Loads from file
# - Sets conversation_id
# - Populates current_conversation
```

**List all**:
```python
conversations = conversation_manager.list_conversations()
# Returns:
# [
#     {
#         "id": "20240115_143022",
#         "started_at": "2024-01-15T14:30:22",
#         "message_count": 5,
#         "first_message": "Hello, I need help with..."
#     }
# ]
```

## Edge Cases

- **Empty conversation**: `started_at` is None in saved file
- **Missing file on load**: Exception raised
- **Corrupted JSON**: Error logged, exception raised
- **No assistant message**: `get_last_assistant_message()` returns None
- **No user message before assistant**: `user_index` and `user_content` are None in pair

## Dependencies

- `settings.conversation_history_dir`: Base path for storage
