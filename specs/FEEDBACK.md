# Feedback System (Experimental)

> **Status**: Experimental - feedback data structure and analysis methods may change

## Overview

Collects user ratings on assistant responses for quality improvement. Supports thumbs up/down and 1-5 star ratings with optional text feedback.

## Key Components

- `src/feedback_system.py`: FeedbackManager class, Feedback dataclass

## Feedback Types

| Type | Values | Positive Threshold |
|------|--------|-------------------|
| `THUMBS_UP` | rating=2 | Always positive |
| `THUMBS_DOWN` | rating=1 | Always negative |
| `RATING_1_5` | 1-5 | 4+ is positive |

## Data Structures

### Feedback
```python
@dataclass
class Feedback:
    id: str                      # Unique identifier
    conversation_id: str         # Links to conversation
    message_index: int           # Position in conversation
    feedback_type: FeedbackType  # Type of feedback
    rating: int                  # Numeric rating
    text_feedback: Optional[str] # Optional improvement suggestions
    timestamp: datetime
    user_query: Optional[str]    # Original question
    assistant_response: Optional[str]  # Rated response
```

### Normalized Score
- `THUMBS_UP` → 1.0
- `THUMBS_DOWN` → 0.0
- `RATING_1_5` → (rating - 1) / 4.0 (maps 1-5 to 0.0-1.0)

## Storage

| File | Contents |
|------|----------|
| `data/feedback/feedback.json` | All feedback entries |

### File Format
```json
{
  "20240115_143022_5_20240201103022": {
    "id": "20240115_143022_5_20240201103022",
    "conversation_id": "20240115_143022",
    "message_index": 5,
    "feedback_type": "thumbs_up",
    "rating": 2,
    "text_feedback": "Very helpful!",
    "timestamp": "2024-02-01T10:30:22",
    "user_query": "How do I...",
    "assistant_response": "You can..."
  }
}
```

## Behavior

### Adding Feedback

```python
# Thumbs up
feedback_manager.add_thumbs_up(
    conversation_id="20240115_143022",
    message_index=5,
    text_feedback="Very helpful!",
    user_query="How do I...",
    assistant_response="You can..."
)

# Thumbs down
feedback_manager.add_thumbs_down(...)

# Star rating
feedback_manager.add_rating(
    ...,
    rating=4  # Must be 1-5
)
```

### Retrieving Feedback

```python
# For specific message
feedback_list = feedback_manager.get_feedback_for_message(
    conversation_id="...",
    message_index=5
)

# For entire conversation
feedback = feedback_manager.get_feedback_for_conversation("...")

# Best/worst responses
best = feedback_manager.get_best_responses(limit=10)
worst = feedback_manager.get_worst_responses(limit=10)

# With text feedback
with_text = feedback_manager.get_responses_with_feedback_text(limit=10)
```

### Statistics

```python
stats = feedback_manager.get_statistics()
# Returns:
# {
#     "total_feedback": 50,
#     "positive_count": 40,
#     "negative_count": 10,
#     "positive_rate": 0.8,
#     "average_score": 0.75,
#     "feedback_with_text": 15,
#     "by_type": {
#         "thumbs_up": {"count": 20, "avg_rating": 2.0},
#         "rating_1_5": {"count": 30, "avg_rating": 3.8}
#     }
# }
```

When there is no feedback, `get_statistics()` omits `positive_rate` and returns zeros for the other fields.

### Few-Shot Examples

```python
examples = feedback_manager.get_few_shot_examples(limit=5)
# Returns:
# [
#     {
#         "user_query": "How do I...",
#         "assistant_response": "You can...",
#         "rating": 5,
#         "normalized_score": 1.0
#     }
# ]
```

Used to provide high-quality examples to agents for learning.

### Improvement Insights

```python
insights = feedback_manager.get_improvement_insights()
# Returns:
# {
#     "common_issues": ["Text from negative feedback..."],
#     "successful_patterns": ["Text from positive feedback..."],
#     "improvement_suggestions": ["All text feedback..."]
# }
```

## Edge Cases

- **Rating out of range**: ValueError for ratings not 1-5
- **No feedback exists**: Statistics return zeros
- **Missing user_query/assistant_response**: Excluded from few-shot examples
- **Empty feedback file**: Returns empty dict, creates on first save

## Dependencies

- `settings.conversation_history_dir.parent`: Base path for feedback storage
