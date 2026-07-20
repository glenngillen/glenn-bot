---
id: glenn-bot-13o
title: Add Memory Decay/Cleanup System
status: open
type: task
priority: 2
created_at: 2025-12-23T15:22:35Z
updated_at: 2026-04-12T14:15:07Z
closed_at: ~
close_reason: ~
external_ref: gh-glenngillen/glenn-bot#5
---
## Problem to Solve

The memory system accumulates data indefinitely without any cleanup mechanism. Over time, this will:
- Cause storage bloat
- Make queries slower as ChromaDB collections grow
- Potentially include outdated or irrelevant information in context

## Current State

- `memory_manager.py` has methods to add memories but no cleanup/decay functions
- No aging or relevance scoring for old memories
- No automated cleanup processes

## Solution Proposal

1. **Memory Aging System**
   - Add timestamp tracking for all memories (some exists already)
   - Implement decay scores based on age and access frequency
   - Add method to get memory age and staleness

2. **Cleanup Mechanisms**
   - Add `cleanup_old_memories(days_threshold)` method
   - Add `prune_low_relevance_memories(threshold)` method
   - Add configuration for automatic cleanup intervals

3. **Memory Consolidation**
   - Add ability to merge similar memories
   - Summarize old conversations before archiving
   - Keep key insights while reducing storage

## Files to Modify

- `src/glenn_bot/memory_manager.py`: Add cleanup methods
- `src/glenn_bot/glenn_agent.py`: Add optional automated cleanup

## Acceptance Criteria

- [ ] Memories have proper aging metadata
- [ ] Cleanup functions available for manual triggering
- [ ] Configurable automatic cleanup option
- [ ] Storage doesn't grow unbounded

## Labels

`enhancement`, `feature`

---
*Issue created via AI analysis of codebase*