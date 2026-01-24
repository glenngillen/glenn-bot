# Context Detection (Experimental)

> **Status**: Experimental - thresholds and behavior may change

## Overview

Automatically detects conversation context from message content and recommends context switches. Uses a hybrid approach combining keyword matching and LLM classification.

## Key Components

- `src/context_detector.py`: ContextDetector class and ContextSwitchResult

## Built-in Context Types

| Context ID | Description | Sample Keywords |
|------------|-------------|-----------------|
| `work` | Professional work, career, business | work, job, project, deadline, meeting, client |
| `personal` | Personal life, family, friends, health | family, friend, health, home, vacation, hobby |
| `creative` | Creative projects, art, writing, music | creative, write, story, art, design, brainstorm |
| `reflection` | Self-reflection, personal growth | reflect, think, growth, meaning, purpose, wisdom |

Sample keywords are illustrative; the actual keyword lists are longer and live in `ContextDetector.CONTEXT_TYPES`.
Only these four contexts are auto-detected; other contexts in the memory system are never recommended by the detector.

## Data Structures

### ContextSwitchResult
```python
@dataclass
class ContextSwitchResult:
    should_switch: bool           # Whether to switch contexts
    recommended_context: str      # Context ID to switch to
    confidence: float             # 0.0 to 1.0
    needs_confirmation: bool      # Whether to ask user
    scores: Dict[str, float]      # All context scores
```

## Thresholds

| Threshold | Default | Purpose |
|-----------|---------|---------|
| `auto_switch_threshold` | 0.8 | Auto-switch without asking |
| `prompt_threshold` | 0.6 | Ask user for confirmation |

Behavior by confidence:
- `>= 0.8`: Auto-switch silently
- `>= 0.6`: Suggest switch, ask for confirmation
- `< 0.6`: No switch recommendation

## Classification Flow

```
Message
    ↓
Keyword Classification
    ↓
Max score >= 0.7? ─── YES ──→ Return keyword scores
    ↓ NO
LLM Classification
    ↓
Blend scores (70% LLM + 30% keywords)
    ↓
Return blended scores
```

## Behavior

### Keyword Classification

1. Convert message to lowercase
2. Count matches for each context's keywords
3. Normalize: `score = min(matches / 5.0, 1.0)`
4. Normalize so all scores sum to 1.0 when there is at least one match; if total is 0, scores remain 0.0

### LLM Classification

Prompt includes:
- Available contexts with descriptions
- Message to classify
- Recent conversation context (if provided)

Expects JSON response:
```json
{"work": 0.2, "personal": 0.1, "creative": 0.6, "reflection": 0.1}
```

Uses `temperature=0.3` for consistent results. If parsing fails, the detector returns a uniform score for all contexts (0.25 each) and blends as usual.

### Switch Decision

```python
result = detector.should_switch_context(current_context_id, scores)
```

1. Find highest-scoring context
2. If already in that context → no switch
3. If score >= auto_switch_threshold → switch (no confirmation)
4. If score >= prompt_threshold → switch (needs confirmation)
5. Otherwise → no switch

## Usage

### Single-Step Detection
```python
result = detector.detect_and_recommend(
    message="I need to finish this report by Friday",
    current_context_id="personal",
    recent_context="Last 5 messages..."
)
# result.should_switch = True
# result.recommended_context = "work"
# result.confidence = 0.85
# result.needs_confirmation = False
```

### Two-Step Detection
```python
scores = detector.classify_message(message, recent_context)
result = detector.should_switch_context(current_context_id, scores)
```

## Edge Cases

- **LLM classification fails**: Falls back to keyword scores only
- **All keyword scores zero + LLM failure**: Returns all-zero scores, so no switch is recommended
- **No scores returned**: Returns `should_switch=False`
- **Already in best context**: Returns current context, no switch
- **Empty message**: May return low scores for all contexts
- **Missing context in LLM response**: Filled with 0.0
- **Recommended context not in memory system**: The main loop skips the switch (no prompt, no auto-switch)

## Dependencies

- `OllamaClient`: For LLM-based classification
