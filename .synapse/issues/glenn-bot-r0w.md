---
id: glenn-bot-r0w
title: Update README - Remove LlamaIndex Reference
status: closed
type: task
priority: 2
created_at: 2025-12-23T15:22:35Z
updated_at: 2026-01-02T05:58:00Z
closed_at: 2026-01-02T05:58:00Z
close_reason: "Completed via PR #15"
external_ref: gh-glenngillen/glenn-bot#4
---
## Problem to Solve

The README mentions LlamaIndex as a key component ("LlamaIndex: For intelligent knowledge retrieval"), but the actual implementation uses BeautifulSoup for web scraping and simple ChromaDB embeddings. LlamaIndex is not used in the codebase.

## Current State

- `requirements.txt` does not include `llama-index`
- `knowledge.py` uses BeautifulSoup4 for URL content extraction
- ChromaDB is used directly without LlamaIndex abstractions

## Solution Proposal

Update the README.md to accurately reflect the actual technology stack:
- Remove LlamaIndex mention
- Add BeautifulSoup4 mention for web scraping
- Clarify ChromaDB is used directly for embeddings

## Acceptance Criteria

- [ ] README accurately reflects actual dependencies
- [ ] Technology descriptions match implementation

## Labels

`documentation`, `bug`

---
*Issue created via AI analysis of codebase*