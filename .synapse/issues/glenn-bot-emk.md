---
id: glenn-bot-emk
title: Add Context Auto-switching
status: closed
type: task
priority: 2
created_at: 2025-12-23T15:22:34Z
updated_at: 2026-04-12T14:15:07Z
closed_at: 2026-01-12T06:47:53Z
close_reason: "Completed via PR #21"
external_ref: gh-glenngillen/glenn-bot#12
---
## Problem to Solve

Users must explicitly set conversation context using `set_context()`. The system doesn't automatically detect what type of conversation is happening and switch contexts accordingly.

## Current State

- `set_context()` method exists for manual context switching
- Contexts: work, personal, creative, reflection
- No automatic detection of conversation topic
- Context persists until manually changed

## Solution Proposal

1. **Intent Classification**
   - Analyze incoming messages to classify intent/topic
   - Use lightweight classifier or LLM call for classification
   - Support gradual confidence thresholds

2. **Auto-switching Logic**
   - Detect when conversation shifts topics
   - Prompt user to confirm context switch
   - Or automatically switch with high confidence

3. **Context Blending**
   - Support mixed contexts (e.g., work-personal crossover)
   - Weight context relevance based on detected topic
   - Remember user preferences for context handling

## Files to Modify

- `src/glenn_bot/conversation.py`: Add auto-detection
- `src/glenn_bot/glenn_agent.py`: Integrate context detection

## Acceptance Criteria

- [ ] System can detect conversation topic
- [ ] Context switches appropriately based on topic
- [ ] User can override auto-switching
- [ ] Optional: Learning user preferences over time

## Labels

`enhancement`, `feature`

---
*Issue created via AI analysis of codebase*