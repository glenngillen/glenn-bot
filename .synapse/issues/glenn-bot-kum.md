---
id: glenn-bot-kum
title: Add Async Operations for Long Tasks
status: closed
type: task
priority: 2
created_at: 2025-12-23T15:22:35Z
updated_at: 2026-04-12T14:15:07Z
closed_at: 2026-01-12T06:41:59Z
close_reason: "Completed via PR #20"
external_ref: gh-glenngillen/glenn-bot#8
---
## Problem to Solve

The codebase imports `asyncio` but doesn't use it. Long-running operations (like scraping multiple URLs, processing large documents) block the main thread and provide poor user experience.

## Current State

- `asyncio` is imported in `glenn_agent.py` but never used
- URL scraping in `knowledge.py` is synchronous
- ChromaDB operations are synchronous
- API endpoints in `api.py` are synchronous

## Solution Proposal

1. **Async Knowledge Processing**
   - Make `fetch_and_store_url` async
   - Support concurrent URL processing
   - Add progress callbacks for long operations

2. **Async API Endpoints**
   - Convert Flask to async (or use FastAPI)
   - Handle streaming responses for long queries
   - Add background task processing

3. **Batch Processing with Progress**
   - Add `batch_process_urls(urls: List[str])` with async processing
   - Return progress updates during processing
   - Support cancellation of long-running tasks

## Files to Modify

- `src/glenn_bot/knowledge.py`: Make URL fetching async
- `src/glenn_bot/api.py`: Add async support
- `src/glenn_bot/glenn_agent.py`: Use async where beneficial

## Acceptance Criteria

- [ ] URL processing doesn't block other operations
- [ ] Progress feedback for long-running tasks
- [ ] Either remove asyncio import or use it properly

## Labels

`enhancement`, `performance`

---
*Issue created via AI analysis of codebase*