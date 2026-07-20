---
id: glenn-bot-ys5
title: Add Batch Document Processing
status: open
type: task
priority: 2
created_at: 2025-12-23T15:22:34Z
updated_at: 2026-04-12T14:15:07Z
closed_at: ~
close_reason: ~
external_ref: gh-glenngillen/glenn-bot#13
---
## Problem to Solve

The knowledge system can only process one URL at a time. There's no way to batch add multiple documents or URLs, making it tedious to populate the knowledge base.

## Current State

- `fetch_and_store_url()` processes single URL
- No batch processing capability
- No file upload support for documents
- Manual one-by-one addition required

## Solution Proposal

1. **Batch URL Processing**
   - Accept list of URLs to process
   - Process concurrently (see async issue #8)
   - Report progress and any failures
   - Support URL file input (one URL per line)

2. **Local Document Support**
   - Support local file uploads (PDF, TXT, MD, HTML)
   - Extract text content from various formats
   - Add to knowledge base with source metadata

3. **Bulk Import Formats**
   - OPML for RSS feeds
   - Browser bookmark exports
   - Spreadsheet of URLs

## Files to Modify

- `src/glenn_bot/knowledge.py`: Add batch processing
- `src/glenn_bot/api.py`: Add batch endpoints
- Add: Document parsing utilities

## Acceptance Criteria

- [ ] Can process multiple URLs in one call
- [ ] Progress reporting during batch operations
- [ ] Error handling for failed items (don't fail entire batch)
- [ ] Support for common document formats

## Labels

`enhancement`, `feature`

---
*Issue created via AI analysis of codebase*