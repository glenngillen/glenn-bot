---
id: glenn-bot-trl
title: Add Knowledge Export/Import
status: closed
type: task
priority: 2
created_at: 2025-12-23T15:22:35Z
updated_at: 2026-04-12T14:15:07Z
closed_at: 2026-01-12T06:41:01Z
close_reason: "Completed via PR #19"
external_ref: gh-glenngillen/glenn-bot#7
---
## Problem to Solve

There's no way to backup, restore, or transfer the knowledge base. If the ChromaDB database gets corrupted or the system needs to be migrated, all accumulated knowledge would be lost.

## Current State

- Knowledge stored in ChromaDB collections
- No export functionality
- No import/restore capability
- No backup mechanism

## Solution Proposal

1. **Export Functionality**
   - Export knowledge base to JSON format
   - Include metadata (source URLs, timestamps, etc.)
   - Support partial exports (by date range, source, etc.)

2. **Import Functionality**
   - Import from JSON backup file
   - Handle duplicates appropriately (skip or update)
   - Validate data before import

3. **Backup Commands**
   - CLI command: `glenn export knowledge --output backup.json`
   - CLI command: `glenn import knowledge --input backup.json`
   - Optional: scheduled automatic backups

## Files to Modify

- `src/glenn_bot/knowledge.py`: Add export/import methods
- `src/glenn_bot/memory_manager.py`: Add collection export helpers
- `src/glenn_bot/cli.py` or similar: Add CLI commands

## Acceptance Criteria

- [ ] Can export entire knowledge base to JSON
- [ ] Can import from JSON backup
- [ ] Export includes all necessary metadata
- [ ] Import handles duplicates gracefully

## Labels

`enhancement`, `feature`

---
*Issue created via AI analysis of codebase*