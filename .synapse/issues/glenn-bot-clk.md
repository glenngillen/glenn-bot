---
id: glenn-bot-clk
title: Add Conversation Search
status: open
type: task
priority: 2
created_at: 2025-12-23T15:22:35Z
updated_at: 2026-04-12T14:15:07Z
closed_at: ~
close_reason: ~
external_ref: gh-glenngillen/glenn-bot#6
---
## Problem to Solve

Currently, conversations can be listed using `list_conversations()`, but there's no way to search through conversation content. Users cannot find specific past discussions without manually browsing through each conversation.

## Current State

- `conversation.py` has `list_conversations()` that returns metadata only
- `memory_manager.py` stores conversations in ChromaDB
- No search functionality exposed to query conversation content

## Solution Proposal

1. **Add Search Method to ConversationManager**
   - `search_conversations(query: str, limit: int = 10)` method
   - Return matching conversations with relevance scores
   - Include snippet of matching content

2. **Add Search Tool for Agent**
   - Create `search_past_conversations` tool
   - Allow Glenn to reference past discussions when relevant

3. **Optional: Full-text Search Enhancement**
   - Add BM25 or hybrid search for better text matching
   - Combine semantic search with keyword search

## Files to Modify

- `src/glenn_bot/conversation.py`: Add search method
- `src/glenn_bot/memory_manager.py`: Add memory search helpers
- `src/glenn_bot/glenn_agent.py`: Add search tool

## Acceptance Criteria

- [ ] Can search conversation content by keywords/phrases
- [ ] Search returns relevant results with context snippets
- [ ] Agent can use search tool to reference past discussions

## Labels

`enhancement`, `feature`

---
*Issue created via AI analysis of codebase*