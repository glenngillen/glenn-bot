---
id: glenn-bot-2k9
title: Add Scheduled Reflection Prompts
status: open
type: task
priority: 2
created_at: 2025-12-23T15:22:34Z
updated_at: 2025-12-23T15:22:34Z
closed_at: ~
close_reason: ~
external_ref: gh-glenngillen/glenn-bot#11
---
## Problem to Solve

The quote reflection system (`get_quote_reflection()`) only works when manually triggered. The README mentions using daily quotes as journal prompts, but there's no automated scheduling system.

## Current State

- `get_quote_reflection()` exists and works
- Must be called manually
- No scheduling or cron-like functionality
- No notification system for daily prompts

## Solution Proposal

1. **Scheduled Task System**
   - Add scheduler for recurring tasks (using APScheduler or similar)
   - Configure reflection prompt times (e.g., morning daily)
   - Support multiple schedule patterns

2. **Notification Integration**
   - Optional email notifications for daily prompts
   - Optional webhook for integration with other systems
   - Store pending prompts for later retrieval

3. **Prompt Customization**
   - Allow custom prompt schedules
   - Support different types of reflections (daily, weekly review)
   - Make prompt topics configurable

## Files to Modify

- `src/glenn_bot/glenn_agent.py`: Add scheduling capability
- Add: `src/glenn_bot/scheduler.py` for task scheduling
- `src/glenn_bot/api.py`: Add endpoints for schedule management

## Acceptance Criteria

- [ ] Can schedule daily reflection prompts
- [ ] Prompts delivered at configured times
- [ ] Flexible schedule configuration
- [ ] Notification system (at least one method)

## Labels

`enhancement`, `feature`

---
*Issue created via AI analysis of codebase*